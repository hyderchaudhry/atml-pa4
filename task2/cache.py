"""Packed pre-question KV payloads. Layout follows [batch, KV head, position, feature]."""

from dataclasses import dataclass
import torch

POLICIES = ("fp16", "recent", "question", "token4", "channel4", "hybrid")
BITS = {"token4": 4, "channel4": 4, "hybrid": 2}
GROUP = 64


def nbytes(x):
    return x.numel() * x.element_size()


def quantize(x, bits, channel):
    # Only the time-aligned complete portion is passed here.
    work = x.transpose(-1, -2).contiguous() if channel else x.contiguous()
    groups = work.reshape(*work.shape[:-1], work.size(-1) // GROUP, GROUP).float()
    offset = groups.amin(-1).half()
    scale = ((groups.amax(-1) - offset.float()) / (2**bits - 1)).clamp_min(2**-24).half()
    codes = (
        ((groups - offset[..., None].float()) / scale[..., None].float())
        .round()
        .clamp(0, 2**bits - 1)
        .byte()
    )
    lanes = 8 // bits
    pieces = codes.reshape(*codes.shape[:-1], GROUP // lanes, lanes).unbind(-1)
    packed = pieces[0].clone()
    for j in range(1, lanes):
        packed.bitwise_or_(pieces[j] << (bits * j))
    return packed, scale, offset


def dequantize(group, bits, channel):
    packed, scale, offset = group
    codes = torch.stack(
        [(packed >> (i * bits)) & (2**bits - 1) for i in range(8 // bits)], -1
    ).flatten(-2)
    x = (codes.float() * scale[..., None].float() + offset[..., None].float()).half().flatten(-2)
    return x.transpose(-1, -2).contiguous() if channel else x


def low_bytes(length, channels, bits):
    blocks, tail = divmod(length, GROUP)
    return channels * (blocks * (GROUP * bits // 8 + 4) + 2 * tail)


def selection(length, channels, policy, scores, device):
    budget = low_bytes(length, channels, 4)
    if policy not in ("recent", "question", "hybrid"):
        return None
    if policy == "hybrid":
        count = max(
            h
            for h in range(72, length + 1)
            if h * (2 * channels + 4) + low_bytes(length - h, channels, 2) <= budget
        )
    else:
        count = min(length, budget // (2 * channels + 4))
    if count < 72:
        raise ValueError("INT4 budget cannot accommodate the fixed safeguards")
    pos = torch.arange(length, device=device)
    middle = pos[8:-64]
    if policy == "recent":
        extra = middle[len(middle) - (count - 72) :]
    else:
        extra = middle[scores[middle].argsort(descending=True, stable=True)[: count - 72]]
    return torch.cat((pos[:8], extra, pos[-64:])).sort().values.int()


@dataclass
class Payload:
    policy: str
    length: int
    index: object
    layers: list

    def accounting(self, channels):
        selected = 0 if self.index is None else self.index.numel()
        high = self.length if self.policy == "fp16" else selected
        blocks, tail = divmod(self.length - high, GROUP) if self.policy in BITS else (0, 0)
        return dict(
            fp16_selected=2 * channels * high,
            indices=4 * selected,
            integer_payload=channels * blocks * GROUP * BITS.get(self.policy, 0) // 8,
            fp16_scales=2 * channels * blocks,
            fp16_offsets=2 * channels * blocks,
            fp16_residual=2 * channels * tail,
        )

    def bytes(self):
        total = 0 if self.index is None else nbytes(self.index)
        for layer in self.layers:
            for high, group, tail in layer:
                total += sum(nbytes(x) for x in (high, tail) if x is not None)
                if group is not None:
                    total += sum(map(nbytes, group))
        return total

    def unpack(self):
        restored = []
        for layer in self.layers:
            pair = []
            for i, (high, group, tail) in enumerate(layer):
                if group is None:
                    x = high if tail is None else tail
                else:
                    x = torch.cat(
                        (dequantize(group, BITS[self.policy], i == 0 and self.policy != "token4"), tail),
                        -2,
                    )
                if self.policy == "hybrid":
                    low = low_indices(self.length, self.index)
                    shape = (*high.shape[:-2], self.length, high.size(-1))
                    full = high.new_empty(shape)
                    full.index_copy_(-2, self.index.long(), high)
                    full.index_copy_(-2, low, x)
                    x = full
                pair.append(x)
            restored.append(tuple(pair))
        return restored


def low_indices(length, high):
    keep = torch.ones(length, device=high.device, dtype=torch.bool)
    keep[high.long()] = False
    return torch.where(keep)[0]


def encode(layers, policy, scores=None, index=None):
    x = layers[0][0]
    b, h, length, d = x.shape
    channels = 2 * len(layers) * b * h * d
    if index is None and policy in ("hybrid", "recent", "question"):
        index = selection(length, channels, policy, scores, x.device)
    low = low_indices(length, index) if policy == "hybrid" else None
    parts = []
    for layer in layers:
        pair = []
        for i, x in enumerate(layer):
            high = x.index_select(-2, index.long()) if index is not None else None
            if policy not in BITS:
                pair.append((x if high is None else high, None, None))
                continue
            values = x.index_select(-2, low) if low is not None else x
            stop = values.size(-2) // 64 * 64
            tail = values[..., stop:, :].clone(memory_format=torch.contiguous_format)
            group = (
                quantize(values[..., :stop, :], BITS[policy], i == 0 and policy != "token4")
                if stop
                else None
            )
            pair.append((high, group, tail))
        parts.append(pair)
    result = Payload(policy, length, index, parts)
    if result.bytes() != sum(result.accounting(channels).values()):
        raise AssertionError("Tensor payload differs from calculated component sizes")
    if policy != "fp16" and result.bytes() > low_bytes(length, channels, 4):
        raise AssertionError("Physical payload exceeds INT4 reference budget")
    return result
