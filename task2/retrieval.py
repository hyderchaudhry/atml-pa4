"""Qwen split-prefill, question-aware scores, cache-policy fidelity and systems measurements."""

import json
import re
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import rotate_half

from common.runtime import read_rows, save_json, sync
from common.setup import DATA, REVISIONS
from .cache import (
    BITS,
    POLICIES,
    encode,
    dequantize,
    low_bytes,
    low_indices,
    nbytes,
    quantize,
    selection,
)
from .measure import attempt, benchmark


def load(device):
    lock = json.loads(REVISIONS.read_text())
    name, rev = lock["repos"]["model_target"], lock["revisions"]["model_target"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev, use_fast=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            name, revision=rev, dtype=torch.float16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    return tok, model


def prompt(tok, case):
    text = case["user_text"]
    chat = tok.apply_chat_template(
        [dict(role="user", content=text)], tokenize=False, add_generation_prompt=True
    )
    offset = chat.index(text)
    encoded = tok(chat, return_offsets_mapping=True, add_special_tokens=False)
    boundary = offset + text.rindex("QUESTION_BEGIN")
    split = next(i for i, (a, b) in enumerate(encoded.offset_mapping) if b > boundary)
    lo, hi = [offset + case[k] for k in ("evidence_char_start", "evidence_char_end")]
    evidence = [i for i, (a, b) in enumerate(encoded.offset_mapping) if a < hi and b > lo]
    assert evidence and max(evidence) < split and chat[lo:hi] == case["evidence_text"]
    return encoded.input_ids, split, evidence


def tensors(cache):
    return [(v.keys, v.values) for v in cache.layers]


def forward(model, ids, cache, start, all_logits=False):
    physical = cache.get_seq_length() if cache is not None else 0
    pos = torch.arange(ids.size(1), device=ids.device)
    return model(
        ids,
        past_key_values=cache,
        use_cache=True,
        position_ids=(pos + start)[None],
        cache_position=pos + physical,
        logits_to_keep=0 if all_logits else 1,
    )


def question_scores(model, layers, suffix, length):
    # Capture normalized input to each SDPA attention module and its actual Qwen RoPE factors.
    scores = []
    handles = []
    for i, block in enumerate(model.model.layers):

        def capture(attn, args, kw, i=i):
            hidden = kw["hidden_states"]
            q = attn.q_proj(hidden).view(1, len(suffix[0]), -1, attn.head_dim).transpose(1, 2)
            cos, sin = kw["position_embeddings"]
            q = (q * cos[:, None] + rotate_half(q) * sin[:, None]).float()
            k = attn.k_proj(hidden).view(1, len(suffix[0]), -1, attn.head_dim).transpose(1, 2)
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            keys = torch.cat((layers[i][0], k), -2).float()
            groups = q.size(1) // keys.size(1)
            logits = q.reshape(1, keys.size(1), groups, q.size(-2), q.size(-1)) @ keys.unsqueeze(
                2
            ).transpose(-1, -2)
            logits *= attn.head_dim**-0.5
            pos = torch.arange(suffix.size(1), device=q.device)
            logits[..., length:].masked_fill_(pos[None, :] > pos[:, None], -torch.inf)
            scores.append(logits.softmax(-1)[..., :length].mean((0, 1, 2, 3)))

        handles.append(block.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        forward(model, suffix, DynamicCache(layers), length)
    finally:
        for hook in handles:
            hook.remove()
    return torch.stack(scores).mean(0)


class Stream:
    """Dense SDPA view of packed KV. Quantize a pending group as soon as 64 positions accumulate."""

    def __init__(self, model, payload, values):
        self.model = model
        self.cache = DynamicCache(values)
        self.absolute = payload.length
        self.policy = payload.policy
        self.device = values[0][0].device
        self.pending = []
        if self.policy in BITS:
            low = (
                low_indices(payload.length, payload.index)
                if self.policy == "hybrid"
                else torch.arange(payload.length, device=self.device)
            )
            self.pending = low[(len(low) // 64) * 64 :].tolist()

    def append(self, ids):
        while ids.size(1):
            n = min(ids.size(1), 64 - len(self.pending)) if self.policy in BITS else ids.size(1)
            out = forward(self.model, ids[:, :n], self.cache, self.absolute)
            if self.policy in BITS:
                self.pending.extend(range(self.absolute, self.absolute + n))
            self.absolute += n
            ids = ids[:, n:]
            if len(self.pending) == 64:
                idx = torch.tensor(self.pending, device=self.device)
                for pair in tensors(self.cache):
                    for i, x in enumerate(pair):
                        channel = i == 0 and self.policy != "token4"
                        block = quantize(x.index_select(-2, idx), BITS[self.policy], channel)
                        x.index_copy_(-2, idx, dequantize(block, BITS[self.policy], channel))
                self.pending.clear()
        return out.logits[:, -1].float()


def measure(fn, device):
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return value, time.perf_counter() - start


def generate(model, tok, payload, values, suffix):
    stream = Stream(model, payload, values)
    logits, suffix_seconds = measure(lambda: stream.append(suffix), values[0][0].device)
    start = time.perf_counter()
    first = logits
    emitted = []
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    for _ in range(16):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite generation logits")
        token = logits.argmax(-1)
        emitted.append(int(token))
        if emitted[-1] in eos or len(emitted) == 16:
            break
        logits = stream.append(token[:, None])
    sync(values[0][0].device)
    decode_seconds = time.perf_counter() - start
    physical = stream.cache.get_seq_length()
    return (
        first,
        emitted,
        tok.decode(emitted, skip_special_tokens=True),
        stream.absolute,
        physical,
        suffix_seconds,
        decode_seconds,
    )


def likelihood(model, payload, values, suffix, answer):
    stream = Stream(model, payload, values)
    logits = stream.append(suffix)
    trajectory = []
    loss = 0.0
    for i, t in enumerate(answer):
        trajectory.append(logits.log_softmax(-1))
        loss += float(-trajectory[-1][0, t])
        if i + 1 < len(answer):
            logits = stream.append(suffix.new_tensor([[t]]))
    return loss, torch.cat(trajectory)


@torch.inference_mode()
def run_case(model, tok, case, directory):
    dest = directory / (case["id"] + ".json")
    if dest.exists():
        return
    device = next(model.parameters()).device
    (ids, length, evidence), prep = measure(lambda: prompt(tok, case), device)
    tokens = torch.tensor([ids], device=device)
    suffix = tokens[:, length:]
    answer = tok.encode(case["code"], add_special_tokens=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full, prefill = measure(lambda: forward(model, tokens[:, :length], None, 0), device)
    layers = tensors(full.past_key_values)
    prefill_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    direct = forward(model, tokens, None, 0)
    original = direct.logits[:, -1].float()
    # Independent ordinary full-prompt greedy continuation for the 100%-retention test.
    ordinary = []
    direct_second = None
    next_logits = original
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos or []
    for i in range(16):
        if not torch.isfinite(next_logits).all():
            raise FloatingPointError("nonfinite direct logits")
        t = next_logits.argmax(-1)
        ordinary.append(int(t))
        if int(t) in eos or i == 15:
            break
        next_logits = (
            forward(model, t[:, None], direct.past_key_values, len(ids) + i).logits[:, -1].float()
        )
        if i == 0:
            direct_second = next_logits
    del direct
    scores, score_time = measure(lambda: question_scores(model, layers, suffix, length), device)
    scoring_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    rows = {}
    channels = sum(x.numel() // length for pair in layers for x in pair)
    full_bytes = 2 * channels * length
    for policy in POLICIES:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        idx, selection_time = measure(
            lambda: selection(length, channels, policy, scores, device), device
        )
        packed, pack_time = measure(lambda: encode(layers, policy, scores, index=idx), device)
        values, unpack_time = measure(packed.unpack, device)
        (logits, emitted, text, end, physical, suffix_time, decode_time), decode = measure(
            lambda: generate(model, tok, packed, values, suffix), device
        )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite post-prefill logits")
        if policy == "fp16":
            reference = logits.log_softmax(-1)
            delta = logits - original
            check = dict(
                max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / original.norm()),
                identical_generation=emitted == ordinary,
                direct_tokens=ordinary,
                passed=bool(delta.norm() / original.norm() < 0.01 and emitted == ordinary),
            )
            identity = torch.arange(length, device=device)
            retained = [tuple(x.index_select(-2, identity) for x in layer) for layer in layers]
            _, identity_ids, *_ = generate(model, tok, packed, retained, suffix)
            check["retain_all_generation_matches_direct"] = identity_ids == ordinary
            check["passed"] &= check["retain_all_generation_matches_direct"]
            del retained
            if direct_second is not None:
                probe = Stream(model, packed, values)
                probe.append(suffix)
                second = probe.append(tokens.new_tensor([[ordinary[0]]]))
                err = second - direct_second
                check.update(
                    decode_max_abs=float(err.abs().max()),
                    decode_relative_l2=float(err.norm() / direct_second.norm()),
                )
                check["passed"] &= check["decode_relative_l2"] < 0.01
                del probe
        drift = float(
            F.kl_div(logits.log_softmax(-1), reference, reduction="batchmean", log_target=True)
        )
        nll, trajectory = likelihood(model, packed, values, suffix, answer)
        if policy == "fp16":
            answer_reference = trajectory
        answer_kl = float(F.kl_div(trajectory, answer_reference, reduction="batchmean", log_target=True))
        idx = packed.index
        selected = idx.tolist() if idx is not None else None
        extracted = re.search(r"(?<!\d)\d{5}(?!\d)", text)
        extra = end - length
        physical_bytes = 2 * channels * physical
        bytes_ = packed.bytes()
        # Explicit high-position map remains fixed. Newly completed low groups replace FP16 tails.
        if policy in BITS:
            high = len(selected or [])
            extra_bytes = low_bytes(length - high + extra, channels, BITS[policy]) - low_bytes(
                length - high, channels, BITS[policy]
            )
        else:
            extra_bytes = extra * channels * 2
        errors = []
        for side in (0, 1):
            if policy in BITS:
                mse = torch.stack(
                    [
                        (orig[side].float() - got[side].float()).square().mean()
                        for orig, got in zip(layers, values)
                    ]
                ).mean()
                errors.append(float(mse))
            else:
                errors.append(None)
        rows[policy] = dict(
            exact_match=bool(extracted and extracted.group() == case["code"]),
            generation=text,
            output_ids=emitted,
            answer_nll=nll,
            answer_token_count=len(answer),
            post_prefill_kl=drift,
            k_mse=errors[0],
            v_mse=errors[1],
            packed_bytes=bytes_,
            storage_components=packed.accounting(channels),
            full_fraction=bytes_ / full_bytes,
            selected_fp16_positions=selected,
            evidence_selected=(
                len(set(evidence) & set(selected)) / len(evidence)
                if selected is not None
                else (1.0 if policy == "fp16" else None)
            ),
            encoded_seconds=pack_time,
            selection_seconds=selection_time,
            teacher_forced_kl=answer_kl,
            unpack_seconds=unpack_time,
            score_seconds=score_time if policy in ("question", "hybrid") else 0.0,
            prefill_seconds=prefill,
            preprocess_seconds=prep,
            decode_including_suffix_seconds=decode,
            suffix_seconds=suffix_time,
            greedy_decode_seconds=decode_time,
            shared_appended_tokens=extra,
            appended_fp16_bytes=extra * channels * 2,
            incremental_packed_bytes=extra_bytes,
            total_live_packed_bytes=bytes_ + extra_bytes,
            execution_dense_cache_bytes=physical_bytes,
            peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        )
        del packed, values
    save_json(
        dest,
        dict(
            id=case["id"],
            code=case["code"],
            position=case["position"],
            distractors=case["distractors"],
            target_tokens=case["target_tokens"],
            actual_tokens=len(ids),
            context_tokens=length,
            suffix_ids=ids[length:],
            evidence_token_positions=evidence,
            evidence_char_span=[case["evidence_char_start"], case["evidence_char_end"]],
            evidence_text=case["evidence_text"],
            prefill_peak_bytes=prefill_peak,
            reference_and_scoring_peak_bytes=scoring_peak,
            reference_check=check,
            budget=low_bytes(length, channels, 4),
            policies=rows,
        ),
    )


@torch.inference_mode()
def systems(model, device, b, length, policy):
    cfg = model.config
    shape = (b, cfg.num_key_value_heads, length, cfg.hidden_size // cfg.num_attention_heads)
    rng = torch.Generator(device=device).manual_seed(6304)
    layers = [
        tuple(torch.randn(shape, generator=rng, device=device, dtype=torch.float16) for _ in range(2))
        for _ in range(cfg.num_hidden_layers)
    ]
    scores = torch.rand(length, generator=rng, device=device)
    token = torch.full((b, 1), 64, device=device, dtype=torch.long)
    pack = lambda: encode(layers, policy, scores)
    encoding = benchmark(pack, device)
    payload = pack()
    decoding = benchmark(payload.unpack, device)
    if payload.index is not None and policy in ("recent", "question"):
        ref = [tuple(x.index_select(-2, payload.index.long()) for x in layer) for layer in layers]
    else:
        ref = layers

    def step(x):
        return forward(model, token, DynamicCache(x), length)

    baseline = benchmark(lambda: step(ref), device)
    with_unpack = benchmark(lambda: step(payload.unpack()), device)
    return dict(
        batch=b,
        length=length,
        policy=policy,
        packed_bytes=payload.bytes(),
        storage_components=payload.accounting(2 * len(layers) * b * shape[1] * shape[-1]),
        dense_reference_bytes=sum(nbytes(x) for pair in ref for x in pair),
        encode=encoding,
        unpack=decoding,
        step_with_unpack=with_unpack,
        step_fp16=baseline,
    )
