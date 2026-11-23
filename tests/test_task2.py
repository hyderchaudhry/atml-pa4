"""Offline Task 2 smoke checks. All evidence is written in temporary directories."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from task2 import cache, exact, recall, retrieval


class Task2Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(6304)

    def test_tiled_output_gradients_and_causality(self):
        q, k, v, g = [torch.randn(2, 4, 23, 8, requires_grad=True) for _ in range(4)]
        expected = exact.outputs(exact.dense, (q, k, v), g)
        for tile in (1, 8, 32):
            actual = exact.outputs(lambda *x: exact.Online.apply(*x, tile), (q, k, v), g)
            for x, y in zip(actual, expected):
                torch.testing.assert_close(x, y, atol=3e-6, rtol=2e-5)
        future = k.detach().clone()
        future[..., 8:, :] += 100
        torch.testing.assert_close(
            exact.Online.apply(q, future, v, 8)[..., :8, :], expected[0][..., :8, :]
        )

    def test_recall_operators(self):
        q, k, v = [torch.randn(1, 4, 137, 32, requires_grad=True) for _ in range(3)]
        weights = (
            (torch.nn.functional.elu(q) + 1) @ (torch.nn.functional.elu(k) + 1).transpose(-1, -2)
        ).tril()
        target = (weights @ v) / (weights.sum(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(recall.linear(q, k, v), target, atol=1e-5, rtol=1e-4)
        local = recall.mask("window", 137, q.device)
        dilated = recall.mask("dilated", 137, q.device)
        self.assertEqual(local[-1].sum().item(), 64)
        self.assertTrue(dilated[3, 136, 8] and not dilated[3, 136, 9])
        for kind in ("dense", "window", "linear", "dilated", "curriculum"):
            out = recall.attend(q, k, v, kind)
            gradients = torch.autograd.grad(out.sum(), (q, k, v))
            self.assertTrue(all(x.isfinite().all() for x in gradients))
        self.assertEqual(3 * (len(recall.RUNS) + 1), 42)

    def test_training_checkpoint_resume(self):
        def fixture(k):
            data = {f: np.zeros((32, 12 if f == "tokens" else k), dtype=np.uint8) for f in recall.FIELDS}
            data["answer_ids"][:] = 128
            data["query_positions"][:] = 8
            return data

        frozen = {
            (k, r): (fixture(k), [f"{k}-{i}" for i in range(32)])
            for k, r in itertools.product((4, 8, 16), recall.REGIMES)
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(recall, "CHUNK", 1), patch.object(
            recall, "CHECKS", (1, 2)
        ), patch.object(recall, "frozen", return_value=frozen), patch.object(
            recall, "training_chunk", side_effect=lambda root, k, *args: fixture(k)
        ):
            root = Path(tmp)
            args = (root, ("dense", 4, "mixed"), 6304, torch.device("cpu"))
            d = recall.run_dir(root, "dense", 4, "mixed", 6304)
            with patch.object(recall, "UPDATES", 1):
                recall.train(*args)
            (d / "result.json").unlink()
            with patch.object(recall, "UPDATES", 2):
                recall.train(*args)
            state = torch.load(d / "checkpoint.pt", weights_only=True)
            self.assertEqual([r[0] for r in state["curve"]], [1, 2])
            self.assertEqual(
                set(json.loads((d / "result.json").read_text())["posthoc_window"]), {"4", "8", "16"}
            )

    def test_real_packing_and_safeguards(self):
        l = 539
        layers = [tuple(torch.randn(2, 2, l, 64).half() for _ in range(2)) for _ in range(2)]
        scores = torch.arange(l).flip(0).float()
        channels = 2 * 2 * 2 * 2 * 64
        budget = cache.low_bytes(l, channels, 4)
        for kind in cache.POLICIES:
            payload = cache.encode(layers, kind, scores)
            values = payload.unpack()
            if kind in ("channel4", "token4"):
                self.assertEqual(payload.bytes(), budget)
                for a, b in zip(layers[0], values[0]):
                    torch.testing.assert_close(a[..., 512:, :], b[..., 512:, :], atol=0, rtol=0)
            if payload.index is not None:
                idx = payload.index.long()
                self.assertTrue(set(range(8)) | set(range(l - 64, l)) <= set(idx.tolist()))
                if kind == "hybrid":
                    torch.testing.assert_close(layers[0][0][..., idx, :], values[0][0][..., idx, :])
                else:
                    torch.testing.assert_close(layers[0][0][..., idx, :], values[0][0])
            if kind != "fp16":
                self.assertLessEqual(payload.bytes(), budget)
        x = torch.arange(64).repeat(1, 2, 64, 1).half()
        for bits in (2, 4):
            levels = (1 << bits) - 1
            x0 = x % (levels + 1)
            packed = cache.quantize(x0, bits, False)
            torch.testing.assert_close(cache.dequantize(packed, bits, False), x0, rtol=0, atol=0)

    def test_qwen_split_and_cache_policies(self):
        # One-character token offsets make the fixture's evidence span independently checkable.
        class Tokenizer:
            def apply_chat_template(self, msg, **kw):
                return "[" + msg[0]["content"] + "]"

            def __call__(self, text, **kw):
                return SimpleNamespace(
                    input_ids=[ord(c) % 100 for c in text],
                    offset_mapping=[(i, i + 1) for i in range(len(text))],
                )

            def encode(self, text, **kw):
                return [ord(c) % 100 for c in text]

            def decode(self, tokens, **kw):
                return "".join(map(str, tokens))

        text = "x" * 40 + "12345" + "y" * 457 + "QUESTION_BEGIN" + "?" * 55
        case = dict(
            id="smoke",
            user_text=text,
            evidence_char_start=40,
            evidence_char_end=45,
            evidence_text="12345",
            code="12345",
            position="start",
            distractors=0,
            target_tokens=2048,
        )
        model = (
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=100,
                    hidden_size=128,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    intermediate_size=192,
                    attn_implementation="sdpa",
                )
            )
            .half()
            .eval()
        )
        with tempfile.TemporaryDirectory() as tmp:
            retrieval.run_case(model, Tokenizer(), case, Path(tmp))
            row = json.loads((Path(tmp) / "smoke.json").read_text())
            self.assertTrue(row["reference_check"]["passed"], row["reference_check"])
            self.assertEqual(set(row["policies"]), set(cache.POLICIES))
            for k, r in row["policies"].items():
                self.assertTrue(np.isfinite(r["teacher_forced_kl"]) and np.isfinite(r["answer_nll"]))
                if k != "fp16":
                    self.assertLessEqual(r["packed_bytes"], row["budget"])
            self.assertTrue(row["policies"]["channel4"]["shared_appended_tokens"] >= 64)


if __name__ == "__main__":
    unittest.main()
