"""Explicit causal attention and an online-softmax reference with recomputed tiled gradients."""

import math
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.runtime import save_json
from .measure import attempt, benchmark

LENGTHS = (256, 512, 1024, 2048, 4096, 8192)


def dense(q, k, v):
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    future = torch.ones(scores.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
    return scores.masked_fill(future, -torch.inf).softmax(-1) @ v


def tile_scores(q, k, i, j):
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if i == j:
        future = torch.ones(s.shape[-2:], device=q.device, dtype=torch.bool).triu(1)
        s.masked_fill_(future, -torch.inf)
    return s


class Online(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, tile):
        qf, kf, vf = (x.float() for x in (q, k, v))
        out, lse = torch.empty_like(qf), torch.empty_like(qf[..., :1])
        for i in range(0, q.size(-2), tile):
            qi = qf[..., i : i + tile, :]
            maximum = torch.full_like(qi[..., :1], -torch.inf)
            total, acc = torch.zeros_like(maximum), torch.zeros_like(qi)
            for j in range(0, i + 1, tile):
                sl = slice(j, j + tile)
                s = tile_scores(qi, kf[..., sl, :], i, j)
                new = torch.maximum(maximum, s.amax(-1, keepdim=True))
                old, p = (maximum - new).exp(), (s - new).exp()
                total = old * total + p.sum(-1, keepdim=True)
                acc = old * acc + p @ vf[..., sl, :]
                maximum = new
            out[..., i : i + tile, :] = acc / total
            lse[..., i : i + tile, :] = maximum + total.log()
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.tile = tile
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse = ctx.saved_tensors
        qf, kf, vf, g = (x.float() for x in (q, k, v, grad))
        dq, dk, dv = (torch.zeros_like(x) for x in (qf, kf, vf))
        tile = ctx.tile
        for i in range(0, q.size(-2), tile):
            a = slice(i, i + tile)
            qi, gi = qf[..., a, :], g[..., a, :]
            delta = (gi * out[..., a, :]).sum(-1, keepdim=True)
            for j in range(0, i + 1, tile):
                b = slice(j, j + tile)
                p = (tile_scores(qi, kf[..., b, :], i, j) - lse[..., a, :]).exp()
                ds = p * (gi @ vf[..., b, :].transpose(-1, -2) - delta) / math.sqrt(q.size(-1))
                dq[..., a, :] += ds @ kf[..., b, :]
                dk[..., b, :] += ds.transpose(-1, -2) @ qi
                dv[..., b, :] += p.transpose(-1, -2) @ gi
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


def errors(actual, expected):
    return {
        key: dict(
            max_abs=float((x.float() - y).abs().max()),
            relative_l2=float((x.float() - y).norm() / y.norm().clamp_min(1e-12)),
        )
        for key, x, y in zip(("output", "dq", "dk", "dv"), actual, expected)
    }


def outputs(fn, inputs, upstream):
    out = fn(*inputs)
    return (out.detach(), *torch.autograd.grad(out, inputs, upstream))


def run_shape(directory, length, dtype, device):
    path = directory / f"{length}_{dtype}.json"
    if path.exists():
        return
    rng = torch.Generator(device=device).manual_seed(6304)
    tensors = [torch.randn(2, 4, length, 64, device=device, generator=rng).to(dtype) for _ in range(4)]
    inputs = [x.requires_grad_() for x in tensors[:3]]
    grad = tensors[3]
    ref = attempt(
        lambda: dict(
            values=outputs(dense, [x.detach().float().requires_grad_() for x in inputs], grad.float())
        )
    )
    expected = ref.pop("values", None)
    rows = {}
    methods = [
        ("dense", dense, None),
        *[(f"tiled_{b}", lambda q, k, v, b=b: Online.apply(q, k, v, b), None) for b in (128, 256)],
    ]
    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ):
        methods.append(
            (
                backend.name,
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                backend,
            )
        )
    for name, fn, backend in methods:
        with sdpa_kernel(backend) if backend else nullcontext():

            def forward():
                with torch.no_grad():
                    fn(*inputs)

            row = dict(
                forward=attempt(lambda: benchmark(forward, device)),
                forward_backward=attempt(lambda: benchmark(lambda: outputs(fn, inputs, grad), device)),
            )
            if expected is not None:
                row["agreement"] = attempt(lambda: errors(outputs(fn, inputs, grad), expected))

            def edge():
                short = [x.detach()[..., :17, :].clone().requires_grad_() for x in inputs]
                with torch.no_grad():
                    short[0].mul_(7)
                    short[1].mul_(7)
                ref_inputs = [x.detach().float().requires_grad_() for x in short]
                g = grad[..., :17, :].contiguous()
                vals = outputs(fn, short, g)
                future = [x.detach().clone() for x in short]
                for x in future[1:]:
                    x[..., 9:, :] += 10
                diff = (fn(*future)[..., :9, :] - vals[0][..., :9, :]).abs().max().item()
                return dict(
                    extreme_scores=errors(vals, outputs(dense, ref_inputs, g.float())),
                    future_max_abs=diff,
                )

            row["edge"] = attempt(edge)
            rows[name] = row
    save_json(
        path,
        dict(
            length=length,
            dtype=str(dtype),
            shape=list(inputs[0].shape),
            device=str(device),
            reference=ref,
            methods=rows,
            seed=6304,
        ),
    )
