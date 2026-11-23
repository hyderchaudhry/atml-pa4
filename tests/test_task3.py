"""Offline Task 3 smoke checks. All generated files live in temporary directories."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common.runtime import read_rows, save_json, save_rows
from task3 import checks, course_banks, data, distill, evaluate, report, sampling


def tiny(device="cpu"):
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            eos_token_id=31,
            pad_token_id=0,
            max_position_embeddings=768,
        )
    )
    return model.eval().to(device=device, dtype=torch.float16).requires_grad_(False)


def training_inputs(root):
    rows = [
        dict(id="math-1", tokens=[2, 3, 4], category="math", workload="math", gold="1"),
        dict(id="short-1", tokens=[1, 8], category="short", workload="familiar", family=1, gold="2"),
    ]
    save_rows(root / "inputs/train.jsonl.gz", rows)
    save_rows(root / "inputs/test.jsonl.gz", rows)
    for r in rows:
        save_json(
            root / "teacher_pool" / (r["id"] + ".json"),
            dict(ids=[3, 4, 5, 6], seed=123, generation_s=0.01),
        )
    return rows


class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(6304)

    def test_markov_and_cached_sampler_paths(self):
        exact = checks.markov()
        self.assertEqual(len(exact["joint"]), 27)
        self.assertEqual(sum(v["observed"] for v in exact["joint"]), 100000)
        self.assertGreaterEqual(exact["p_value"], 0.01)
        self.assertLess(checks.markov(incorrect=True)["p_value"], 0.01)
        cases = checks.sampler_paths()
        self.assertEqual(len(cases), 7)
        self.assertEqual(cases["fully_accepted_bonus"]["rounds"][0]["draft_passes"], 3)
        self.assertTrue(cases["bonus_eos"]["eos"])
        self.assertTrue(cases["one_token_cap"]["truncated"])

    def test_real_cache_padding_and_exact_equivalence(self):
        model = tiny()
        prompts = [[2, 5], [3, 9, 1, 4, 7]]
        self.assertTrue(checks.cache_check(model, prompts[1])["passed"])
        outputs = sampling.batches(model, prompts, [457, 458], 7, {31}, 0)
        for p, seed, out in zip(prompts, [457, 458], outputs):
            self.assertEqual(sampling.batches(model, [p], [seed], 7, {31}, 0)[0], out)
        identical = sampling.decode(model, prompts[0], 7, set(), 101, model, 2)
        self.assertEqual([r["accepted"] for r in identical["rounds"]], [2, 2, 1])
        self.assertEqual(identical["target_decode_passes"], 3)
        self.assertEqual(identical["draft_decode_passes"], 7)

    def test_instructor_generation_rng(self):
        model = tiny()
        x = torch.tensor([[2, 5, 4]])
        opts = dict(max_new_tokens=5, do_sample=True, top_k=0, top_p=1.0, temperature=1.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            control = model.generate(x, **opts)
        before = torch.random.get_rng_state()
        with course_banks.course_generation_rng():
            result = model.generate(x, generator=torch.Generator().manual_seed(123), **opts)
        self.assertTrue(torch.equal(control, result))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_token_exact_distillation_reload_and_merge(self):
        device = torch.device("cpu")
        base, teacher = tiny(), tiny()
        with tempfile.TemporaryDirectory() as td, patch.object(
            distill, "load_model", side_effect=lambda *a: copy.deepcopy(base)
        ), patch.object(distill, "CAPS", {"math": 4, "short": 4}):
            root = Path(td)
            training_inputs(root)
            for method in ("off", "on"):
                name = method + "_6304"
                d = root / "students" / name
                distill.train(root, name, teacher, device, 0, {"math": 6, "short": 2}, 5)
                record = json.loads((d / "training.json").read_text())
                self.assertEqual(record["counts"], {"math": 6, "short": 2})
                self.assertEqual([r["tokens"] for r in record["updates"]], [5, 3])
                self.assertEqual(sum(r["scored"] for r in record["presentations"]), 8)
                self.assertEqual("student_rollout_s" in record["timings"], method == "on")
                self.assertTrue(all(r["forward_kl"] > 0 for r in record["updates"]))
                self.assertTrue(any(r["quota_truncated"] for r in record["presentations"]))
                ck = torch.load(d / "resume.pt", weights_only=True)
                self.assertEqual({p.dtype for p in ck["adapter"].values()}, {torch.float32})
                self.assertEqual(
                    {v["exp_avg"].dtype for v in ck["optimizer"]["state"].values()}, {torch.float32}
                )
                self.assertEqual(len(ck["adapter"]), 8)
                self.assertTrue(
                    any(p.abs().max() > 0 for n, p in ck["adapter"].items() if n.endswith(".b"))
                )
                (d / "training.json").unlink()
                distill.train(root, name, teacher, device, 0, {"math": 6, "short": 2}, 5)
                self.assertEqual(record, json.loads((d / "training.json").read_text()))
                distill.merge_checkpoint(root, name, device)
                self.assertLess(json.loads((d / "merge.json").read_text())["relative_l2"], 0.02)
                loaded = Qwen2ForCausalLM.from_pretrained(d / "merged", dtype=torch.float16)
                self.assertFalse(
                    torch.equal(
                        loaded.model.layers[0].self_attn.o_proj.weight,
                        base.model.layers[0].self_attn.o_proj.weight,
                    )
                )

    def test_prefix_alignment_and_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = training_inputs(root)
            model = tiny()
            course_rows = []
            for index, r in enumerate(rows):
                r["text"] = r["id"]
                for bank in evaluate.BANKS[:2]:
                    for pos in data.positions(8, r["id"], bank):
                        course_rows.append(
                            dict(
                                bank=bank,
                                prompt_id=r["id"],
                                prompt=r["text"],
                                position=pos,
                                response_prefix_token_ids=[2] * (pos - 1),
                                source_length=8,
                                source_generation_seed=6304
                                + 10007 * index
                                + (0 if bank == "teacher" else 123),
                            )
                        )
            source = root / "course.jsonl.gz"
            save_rows(source, course_rows)
            evaluate.supplied_banks(root, rows, source)
            for name in data.DRAFTS[1:]:
                save_json(root / "students" / name / "merge.json", dict(adapter_sha256="toy"))
            model.generation_config.repetition_penalty = 1.1
            with patch.object(evaluate, "student", return_value=model):
                evaluate.build_banks(root, model, rows, torch.device("cpu"), 0, 1, 0)
                evaluate.prefix_metrics(root, model, rows, torch.device("cpu"), 0, 1)
            alignment = report.prefix_report(root, rows)
            self.assertEqual({r["producer"] for r in alignment}, set(evaluate.BANKS))
            self.assertTrue(all(abs(r["expected_acceptance"] - 1) < 1e-6 for r in alignment))
            for r in rows:
                self.assertEqual(
                    {v["draft"] for v in read_rows(root / "prefix_metrics" / (r["id"] + ".jsonl.gz"))},
                    set(data.DRAFTS),
                )
            self.assertEqual(len(data.positions(192, "example", "teacher")), 48)
            self.assertTrue(evaluate.assess(rows[0], "x = 7.\n#### 1")["correct"])
            self.assertFalse(evaluate.assess(rows[0], "unfinished")["correct"])
            self.assertFalse(evaluate.assess(rows[1], "2 crates")["format_compliant"])
            self.assertTrue(evaluate.assess(rows[1], "2")["correct"])

    def test_speed_and_quality_evidence(self):
        from types import SimpleNamespace

        device = torch.device("cpu")
        tok = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: "2")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = training_inputs(root)[1:]
            model = tiny()
            # Common near-certain EOS keeps the smoke fast while exercising both real timing paths.
            eos = set(range(32))
            with patch.object(evaluate, "student", return_value=model), patch.object(
                evaluate, "eos_ids", return_value=eos
            ):
                evaluate.execution(root, model, rows, device, 0, 1, tok, baseline=True)
                evaluate.execution(root, model, rows, device, 0, 1, tok)
                evaluate.quality(root, rows, device, 0, 1, tok, eos)
            alignment = [
                dict(workload=w, draft=d, producer=p, position="short", expected_acceptance=1.0)
                for w in ("all", "familiar")
                for d in data.DRAFTS
                for p in evaluate.BANKS
            ]
            result = report.execution_report(root, {}, alignment, rows)
            self.assertTrue(all(r["accepted_examined"] == 1 for r in result))
            with (root / "per_prompt_speed.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 24)
            self.assertEqual(len(list((root / "quality").glob("*/*.json"))), 10)
            # Feed deterministic fixtures to the final evidence collector in addition to the measured toy runs.
            save_rows(root / "inputs/test.jsonl.gz", rows)
            for name in data.DRAFTS[1:]:
                d = root / "students" / name
                save_json(
                    d / "training.json",
                    dict(
                        counts=data.QUOTAS,
                        updates=[dict(tokens=40000, scored=40000, forward_kl=0.25, peak_bytes=1)],
                        timings={},
                        training_s=1,
                        base_parameters=8,
                        teacher_parameters=16,
                        adapter_parameters=2,
                        overflow_skips=0,
                    ),
                )
                save_json(d / "merge.json", dict(merge_s=0.1, relative_l2=0.001))
            records = []
            for bank in evaluate.BANKS:
                save_json(evaluate.bank_path(root, bank, rows[0]), dict(positions=[1], length=1))
                for name in data.DRAFTS:
                    records.append(
                        dict(
                            id=rows[0]["id"],
                            workload="familiar",
                            producer=bank,
                            draft=name,
                            position=1,
                            bin="1-16",
                            forward_kl=0.0,
                            reverse_kl=0.0,
                            tv=0.0,
                            expected_acceptance=1.0,
                            agreement=1,
                        )
                    )
            save_rows(root / "prefix_metrics" / (rows[0]["id"] + ".jsonl.gz"), records)
            report.generate(root)
            for file in (
                "loss.png",
                "seed_dispersion.csv",
                "standalone_accuracy.csv",
                "target_spec_consistency.json",
                "paired_method_differences.csv",
                "aligned_continuations.json",
                "training.csv",
            ):
                self.assertGreater((root / file).stat().st_size, 0)
            self.assertEqual(report.cost_model(1, 1, 2, 2), (0.75, None))
            self.assertAlmostEqual(report.cost_model(0.5, 0.5, 1, 2)[0], 0.875)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_both_cuda_devices(self):
        from concurrent.futures import ThreadPoolExecutor

        def check(i):
            with torch.cuda.device(i):
                model = tiny(f"cuda:{i}")
                args = (model, [1, 2, 3], 5, set(), 6304, model, 2)
                run = sampling.decode(*args, profile=True)
                return checks.cache_check(model, [1, 2, 3, 4])["passed"] and len(run["ids"]) == 5

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(check, [0, 1])), [True, True])


if __name__ == "__main__":
    unittest.main()
