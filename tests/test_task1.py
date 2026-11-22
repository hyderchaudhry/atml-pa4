"""Offline smoke tests with a randomly initialized miniature Qwen. No downloads or persistent results."""

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.runtime import read_rows, seed_all
from task1.calibration import collect
from task1.evaluation import format_example, token_losses
from task1.protocol import CONDITIONS, PROJECTIONS
from task1.pruning import allocate, prune
from task1.quantization import channel_inverse, group_quantize, mixed_bits, producer_inverse, storage
from task1.reporting import paired_interval, summarize
from task1.run import execute, frozen_plan
from task1.study import Study


class CompressionSmoke(unittest.TestCase):
    def test_masks_and_budgets(self):
        w = torch.tensor([[1.0, -4.0, 2.0, 3.0], [3.0, 1.0, 8.0, 2.0]])
        norm = torch.tensor([8.0, 1.0, 1.0, 1.0])
        self.assertEqual(prune(w, 4).tolist(), [[0.0, -4.0, 0.0, 3.0], [3.0, 0.0, 8.0, 0.0]])
        self.assertEqual(prune(w, 4, norm, True).tolist(), [[1.0, -4.0, 0.0, 0.0], [3.0, 0.0, 8.0, 0.0]])
        sizes = {"x": 625, "y": 119, "z": 608}
        for target in (50, 75):
            alloc = allocate(sizes, dict(x=0.1, y=10.0, z=0.8), target, 0.5)
            self.assertEqual(sum(alloc.values()), sum(round(n * target / 100) for n in sizes.values()))
            self.assertTrue(all(0.1 <= alloc[k] / n <= 0.85 for k, n in sizes.items()))

    def test_quantization(self):
        w = torch.tensor([[-2.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(group_quantize(w, 2), w)
        for b in (2, 3, 4, 8):
            for v in (-2.0, 0.0, 3.0):
                x = torch.full((3, 129), v)
                torch.testing.assert_close(group_quantize(x, b), x, atol=0.002, rtol=0.001)
            parts = storage((3, 129), b, True)
            self.assertEqual(sum(parts.values()), 6 * 128 * b // 8 + 6 * 3 + 2 * 129)
        costs = [{2: 20, 3: 30, 4: 40, 8: 80}] * 3
        errors = [{b: (i + 1) ** 3 / b for b in costs[i]} for i in range(3)]
        result = mixed_bits(costs, errors, 90)
        feasible = [
            bits
            for bits in itertools.product(costs[0], repeat=3)
            if len(set(bits)) > 1 and sum(bits) * 10 <= 90
        ]

        def score(bits):
            return sum(e[b] for e, b in zip(errors, bits))

        self.assertEqual(score(result), min(map(score, feasible)))

    @torch.inference_mode()
    def test_graph_compensation(self):
        seed_all(6304)
        model = (
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=48,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    num_hidden_layers=1,
                )
            )
            .float()
            .eval()
        )
        block = model.model.layers[0]
        block.self_attn.v_proj.bias.uniform_(-1.0, 1.0)
        ids = torch.tensor([[2, 5, 3, 8, 4]])
        reference = model(ids).logits.clone()
        for name, source, receiver in [
            ("o_proj", block.self_attn.v_proj, block.self_attn.o_proj),
            ("down_proj", block.mlp.up_proj, block.mlp.down_proj),
        ]:
            inv = channel_inverse(torch.rand(receiver.in_features), 0.5, name, model.config).float()
            fold = producer_inverse(inv, name, model.config)
            receiver.weight.div_(inv)
            source.weight.mul_(fold[:, None])
            if source.bias is not None:
                source.bias.mul_(fold)
        torch.testing.assert_close(model(ids).logits, reference, atol=1e-6, rtol=1e-5)

    def test_condition_coverage(self):
        self.assertEqual(len(CONDITIONS), 18)
        full = {c.name for c in CONDITIONS if c.full_hellaswag}
        self.assertEqual(
            full, {"fp16", "p1_50", "p2_50", "p3_50", "q1_3", "q1_4", "q2_3", "q2_4", "q3_3", "q3_4"}
        )

    @torch.inference_mode()
    def check_parallel_scoring(self, first_device, second_device):
        torch.set_num_threads(1)
        seed_all(6304)
        cfg = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
        )
        model = Qwen2ForCausalLM(cfg).half().eval().to(first_device)
        replica = copy.deepcopy(model).to(second_device)
        windows = [{"source_row": i, "token_offset": 0, "token_ids": [2, i + 1, 3, 8]} for i in range(2)]
        study = Study(model, collect(model, windows, [1]))
        seqs = [[2, 3, choice] for choice in range(4)]
        items = [
            dict(id=i, choices=["a", "b", "c", "d"], gold=i % 4, prefix=2, ids=seqs) for i in range(131)
        ]
        suites = {"wikitext2_test": windows, "c4_validation": windows}
        c = next(c for c in CONDITIONS if c.name == "p2_50")
        with tempfile.TemporaryDirectory() as temp:
            single, dual = (Path(temp) / name for name in ("single", "dual"))
            kwargs = dict(bench_kwargs=dict(lengths=(3,), steps=1, reps=1, warmups=1))
            args = (study, c, {}, suites, items, [1, 2, 3])
            execute(*args, single, **kwargs)
            # Interrupt after scoring to exercise resuming the two workers' saved shards.
            with patch("task1.run.benchmark", side_effect=RuntimeError("synthetic interruption")):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    execute(*args, dual, replica=replica, **kwargs)
            with patch("task1.quality.hellaswag", side_effect=AssertionError("unexpected re-scoring")):
                execute(*args, dual, replica=replica, **kwargs)
            for suite in (*suites, "hellaswag"):
                path = Path(c.name) / (suite + ".jsonl.gz")
                a, b = (read_rows(d / path) for d in (single, dual))
                self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
                key = "scores" if suite == "hellaswag" else "nll_sum"
                np.testing.assert_allclose(
                    [r[key] for r in a], [r[key] for r in b], rtol=1e-4, atol=1e-4
                )
            for source, dest in zip(model.parameters(), replica.parameters()):
                torch.testing.assert_close(source.cpu(), dest.cpu(), rtol=0, atol=0)
            result = json.loads((dual / c.name / "result.json").read_text())
            self.assertEqual(result["quality"]["hellaswag"]["n"], len(items))
            self.assertEqual(result["benchmark_device"], first_device)
            self.assertEqual(set(result["quality_peak_bytes_by_device"]), {first_device, second_device})

    def test_two_worker_scoring_and_resume(self):
        self.check_parallel_scoring("cpu", "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_two_cuda_devices(self):
        self.check_parallel_scoring("cuda:0", "cuda:1")

    @torch.inference_mode()
    def test_pipeline(self):
        torch.set_num_threads(1)
        seed_all(6304)
        cfg = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            attn_implementation="sdpa",
        )
        model = Qwen2ForCausalLM(cfg).eval().half()
        windows = [
            {"source_row": i, "token_offset": 0, "token_ids": list(range(i + 1, i + 1 + n))}
            for i, n in enumerate((10, 7, 9, 6))
        ]
        q, endings, gold = format_example(
            {
                "example": {
                    "activity_label": "Demo",
                    "ctx_a": "Mix.",
                    "ctx_b": "then STIR",
                    "endings": ["One", "Two", "Three", "Four"],
                    "label": "1",
                }
            }
        )
        self.assertEqual(q, "Demo: Mix. Then stir")
        seqs = [[3, 7, 1], [3, 7, 5, 4], [3, 7, 8, 9, 3], [3, 7, 2]]
        items = [dict(id=i, context=q, choices=endings, gold=gold, prefix=2, ids=seqs) for i in range(3)]
        actual = token_losses(model, seqs, [2] * 4, max_logit_rows=2)
        for seq, score in zip(seqs, actual):
            ids = torch.tensor([seq])
            logits = model(ids[:, :-1]).logits[:, 1:].float().log_softmax(-1)
            expected = -logits.gather(-1, ids[:, 2:, None]).sum().item()
            self.assertAlmostEqual(score, expected, places=3)
        calibration = [dict(row, token_ids=row["token_ids"][:6]) for row in windows]
        stats = collect(model, calibration, [0, 3])
        self.assertEqual(tuple(stats["0.q_proj"]["sample"].shape), (8, 32))
        study = Study(model, stats)
        self.assertEqual(len(study.modules), 3 * len(PROJECTIONS))
        self.assertEqual(study.dense_bytes, sum(p.numel() * 2 for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            plan = frozen_plan(study, windows, out)
            self.assertEqual(frozen_plan(study, windows, out)["mixed"], plan["mixed"])
            for c in CONDITIONS:
                execute(
                    study,
                    c,
                    plan,
                    {"wikitext2_test": windows, "c4_validation": windows},
                    items,
                    windows[0]["token_ids"],
                    out,
                    dict(lengths=(4, 8), steps=2, reps=1, warmups=1),
                )
                r = json.loads((out / c.name / "result.json").read_text())
                self.assertTrue(np.isfinite(r["quality"]["c4_validation"]["ppl"]))
                self.assertEqual(
                    r["quality"]["c4_validation"]["n"], sum(len(w["token_ids"]) - 1 for w in windows)
                )
                if c.family == "q3":
                    other = json.loads((out / f"q2_{c.target}" / "result.json").read_text())
                    self.assertLessEqual(
                        r["resources"]["packed_model_bytes"], other["resources"]["packed_model_bytes"]
                    )
                    self.assertGreater(len(set(plan["mixed"][str(c.target)])), 1)
                if c.family in ("p1", "p2", "p3", "wanda_2of4"):
                    self.assertEqual(1 - r["resources"]["effective_nonzero_fraction"], c.target / 100)
            summarize(out, items)
            self.assertEqual(len(json.loads((out / "paired_intervals.json").read_text())), 4)
            self.assertTrue((out / "allocations.png").stat().st_size > 0)
            reference = read_rows(out / "fp16/wikitext2_test.jsonl.gz")
            self.assertEqual(paired_interval(reference, reference, "ppl", draws=15)["high"], 0.0)
            study.restore()
            for k, mod in study.modules.items():
                self.assertTrue(torch.equal(mod.weight, study.original[k][0]))
                if mod.bias is not None:
                    self.assertTrue(torch.equal(mod.bias, study.original[k][1]))


if __name__ == "__main__":
    unittest.main()
