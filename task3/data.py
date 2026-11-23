"""Pinned course rows and shared non-thinking Qwen chat formatting."""

import hashlib
import json
import random
import subprocess
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.runtime import digest, read_rows, save_rows
from common.setup import DATA, DATA_COMMIT, DATA_URL, REVISIONS

LOCK = json.loads(REVISIONS.read_text())
SEEDS = (6304, 6305)
DRAFTS = ("original", *(f"{m}_{s}" for s in SEEDS for m in ("off", "on")))
QUOTAS = {"math": 30000, "short": 10000}
CAPS = {"math": 192, "short": 32}


def derived(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "big") % (2**63 - 1)


def tokenizer(key):
    return AutoTokenizer.from_pretrained(
        LOCK["repos"][key], revision=LOCK["revisions"][key], use_fast=True
    )


def load_model(key, device, path=None):
    source = LOCK["repos"][key] if path is None else str(path)
    model = (
        AutoModelForCausalLM.from_pretrained(
            source,
            revision=LOCK["revisions"][key] if path is None else None,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    return model.requires_grad_(False)


def eos_ids(model):
    eos = model.generation_config.eos_token_id
    return {eos} if isinstance(eos, int) else set(eos)


def chat(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
    )


def prepare(root):
    if not DATA.exists():
        subprocess.run(["git", "clone", DATA_URL, str(DATA)], check=True)
        subprocess.run(["git", "-C", str(DATA), "checkout", "--detach", DATA_COMMIT], check=True)
    manifest = json.loads(
        subprocess.check_output(["git", "-C", str(DATA), "show", f"{DATA_COMMIT}:SHA256SUMS.json"])
    )
    target, draft = tokenizer("model_target"), tokenizer("model_draft")
    if target.get_vocab() != draft.get_vocab():
        raise ValueError("Target and draft token vocabularies differ")
    all_text = set()
    for split in ("train", "test"):
        shortpath = DATA / f"data/short_form/short_{split}.jsonl"
        idpath = DATA / f"data/public_manifests/gsm8k_{split}_ids.jsonl"
        for path in (shortpath, idpath):
            if digest(path) != manifest[str(path.relative_to(DATA))]:
                raise ValueError(f"Changed course input: {path}")
        ids = read_rows(idpath)
        prepared = DATA / f"data/public_locked/gsm8k_{split}_locked.jsonl.gz"
        revision_file = DATA / "data/local_source_revisions.json"
        if (
            prepared.exists()
            and revision_file.exists()
            and json.loads(revision_file.read_text()) == LOCK
        ):
            records = read_rows(prepared)
            if [(r["order"], r["source_row_index"]) for r in records] != [
                (r["order"], r["row_index"]) for r in ids
            ]:
                raise ValueError("GSM8K row manifest mismatch")
            examples = [r["example"] for r in records]
            for r in records:
                content = json.dumps(r["example"], sort_keys=True, default=str).encode()
                if hashlib.sha256(content).hexdigest() != r["source_sha256"]:
                    raise ValueError("Prepared GSM8K example checksum mismatch")
        else:
            from datasets import load_dataset

            ds = load_dataset(
                LOCK["repos"]["gsm8k"], "main", split=split, revision=LOCK["revisions"]["gsm8k"]
            )
            examples = [ds[i["row_index"]] for i in ids]
        rows = []
        for r, ex in zip(ids, examples):
            ident = f"GSM8K-{r['order']:04d}" if split == "test" else f"GSM8K-train-{r['order']:04d}"
            rows.append(
                dict(
                    id=ident,
                    source_row=r["row_index"],
                    text=ex["question"],
                    gold=ex["answer"].split("####")[-1].strip(),
                    category="math",
                    workload="math",
                )
            )
        for r in read_rows(shortpath):
            rows.append(
                dict(
                    id=r["id"],
                    text=r["input"],
                    gold=r["answer"],
                    family=r["family"],
                    category="short",
                    workload=r["split"],
                )
            )
        counts = Counter(r["workload"] for r in rows)
        if counts != (
            {"math": 500, "train": 300}
            if split == "train"
            else {"math": 60, "familiar": 30, "shifted": 30}
        ):
            raise ValueError(f"Incorrect course split counts: {counts}")
        texts = {r["text"] for r in rows}
        if texts & all_text:
            raise ValueError("Train/test prompt overlap")
        all_text.update(texts)
        for r in rows:
            r["tokens"] = chat(target, r["text"])
            if r["tokens"] != chat(draft, r["text"]):
                raise ValueError("Chat-template mismatch")
        save_rows(root / f"inputs/{split}.jsonl.gz", rows)


def positions(length, prompt_id, bank):
    rng = random.Random(f"PA4-banks-v1:6304:{prompt_id}:{bank}")
    out = []
    for lo, hi, limit in ((1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)):
        candidates = list(range(lo, min(length, hi) + 1))
        out.extend(sorted(rng.sample(candidates, min(limit, len(candidates)))))
    return out
