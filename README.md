# ATML PA4 · Tasks 1, 2 and 3

Run the model-compression study on one or two CUDA GPUs. The runner covers 18 conditions and writes evidence to `results/task1/`. Shared setup and experiment bookkeeping live in `common/`. Task 2 has a separate [Kaggle workflow](#run-task-2-on-kaggle) and writes to `results/task2/`. [Run Task 3 on Kaggle](#run-task-3-on-kaggle) for draft-model distillation and exact speculative decoding. Its evidence goes in `results/task3/`.

## Run Task 1 on Kaggle

1. Enable **Internet** and choose a two-GPU accelerator in the notebook settings for parallel scoring. Use the course environment lock if you have it. This checkout includes the course data repository’s dependency constraints. A separate full course environment lock was unavailable. The runner keeps Kaggle’s CUDA-enabled PyTorch, enforces Transformers 4.57.1 and huggingface_hub 0.36.2, and records the installed environment.
2. Run cells 1 and 2 for the Task 1 smoke tests. Continue with cells 3 and 4 when you want to run the full experiment.
3. Save `/kaggle/working/pa4/results/task1/` as Kaggle output. To continue a saved run in a new session, copy that directory, including its `cache/` subfolder and partial condition files, back to the same writable location after cloning. Use the same code, data, GPU type, and package versions for matched runs.

**Cell 1 · Clone and enter your repository**

```python
from pathlib import Path
import os
import subprocess
import sys

repo = Path("/kaggle/working/pa4")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/hyderchaudhry/atml-pa4.git", str(repo)], check=True)
os.chdir(repo)

def run(*args):
    subprocess.run([sys.executable, *args], check=True)
```

**Cell 2 · Install dependencies and run the Task 1 smoke tests**

```python
run("-m", "pip", "install", "-r", "requirements.txt")
run("-m", "unittest", "discover", "-s", "tests", "-p", "test_task1.py", "-v")
```

The tests generate their own inputs and miniature Qwen weights. They run on CPU, with an additional test when two CUDA devices are visible. You can stop here after the tests pass.

**Cell 3 · Prepare the course data for the full experiment**

The assignment repo ignores `datasets/`. On Kaggle, the cell below creates a separate checkout of the instructor’s repo at the exact commit required by Task 1. Your local data-only commit can stay on your machine. To reuse its prepared files and save the C4 download, upload the contents of your local `datasets/data/public_locked/` as a Kaggle input and set `prepared_assets` to the mounted directory containing all seven `.jsonl.gz` files. Leave it as `None` to generate them from the saved revisions.

The pinned instructor commit deleted `scripts/check_release_STAFF_LEGACY.py` but kept its checksum in `SHA256SUMS.json`. This causes `python -m common.setup` to stop at verification. The cell checks every other manifest entry, then validates the public-data counts and all five Task 1 asset hashes.

```python
import hashlib
import json
import shutil
from common.setup import DATA, DATA_COMMIT, DATA_URL, REVISIONS, task1_hashes

prepared_assets = None  # Or Path("/kaggle/input/YOUR-DATA/public_locked")

if not DATA.exists():
    subprocess.run(["git", "clone", DATA_URL, str(DATA)], check=True)
    subprocess.run(["git", "-C", str(DATA), "checkout", "--detach", DATA_COMMIT], check=True)
head = subprocess.check_output(["git", "-C", str(DATA), "rev-parse", "HEAD"], text=True).strip()
if head != DATA_COMMIT:
    raise ValueError(f"Use a fresh Kaggle assignment clone. Expected {DATA_COMMIT}, found {head}.")

checks = json.loads((DATA / "SHA256SUMS.json").read_text())
del checks["scripts/check_release_STAFF_LEGACY.py"]
for name, expected in checks.items():
    if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Course file failed verification: {name}")

lock = DATA / "data/local_source_revisions.json"
if lock.exists() and json.loads(lock.read_text()) != json.loads(REVISIONS.read_text()):
    raise ValueError("Existing data uses different source revisions. Use a fresh Kaggle assignment clone.")
shutil.copyfile(REVISIONS, lock)
locked = DATA / "data/public_locked"
if prepared_assets is None:
    run(str(DATA / "scripts/setup_data.py"), "--public")
else:
    task1_hashes(prepared_assets)
    locked.mkdir(exist_ok=True)
    for path in prepared_assets.glob("*_locked.jsonl.gz"):
        shutil.copyfile(path, locked / path.name)

run(str(DATA / "scripts/check_setup.py"))
print("Verified Task 1 assets:", len(task1_hashes(locked)))  # Expected: 5
```

To run the instructor’s dataset tests too, use `run("-m", "unittest", "discover", "-s", "datasets/tests", "-v")` after cell 3.

**Cell 4 · Run the full study when ready**

```python
import torch
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
run("-m", "task1.run", "--gpus", "2")
```

With two GPUs, GPU 0 scores WikiText-2 while GPU 1 scores C4. Both then work through alternating 128-example HellaSwag shards as each finishes its corpus. Each GPU holds a complete copy of the current condition. The runner merges scores in manifest order, including after a restart. Calibration, allocation selection and compression run on GPU 0. After scoring finishes on both devices, GPU 0 measures every prefill and decode latency with GPU 1 idle.

Use `--gpus 1` for a single-GPU session. Use the same GPU count for every invocation sharing an output directory, including `--prepare-only`. Each condition records peak scoring memory per device, summed HellaSwag shard time, elapsed scoring time for the current invocation, and the device used for latency measurements.

For a session split, freeze the plan first and then run one or more conditions. The runner includes the FP16 reference automatically. A later unrestricted invocation fills in the remaining conditions.

```bash
python -m task1.run --gpus 2 --prepare-only
python -m task1.run --gpus 2 --conditions p2_50 p3_50 q2_4 q3_4
python -m task1.run --gpus 2
```

A completed condition has an atomically written `result.json`. The runner skips completed conditions, reuses full-calibration statistics and the frozen plan, and checkpoints HellaSwag every 128 examples. A failed condition records the phase and error in its own `status.json`. Preserve the entire output directory when moving between sessions. The same commands accept `--output PATH` for a separate run.

## Fixed protocol

The reference seed is **6304**. Every condition starts from the original, tied FP16 Qwen/Qwen2.5-1.5B base checkpoint at revision `8faed761d45a263340a0528343f099c05c9a4323`. The model stays in inference mode with SDPA. A CPU snapshot restores the eligible weights and any folded FP16 biases between conditions. The seven eligible projections per block are Q, K, V, O, gate, up and down. The inventory counts shared embeddings and the output head once.

| Condition | Targets | Rule |
| --- | --- | --- |
| FP16 | Dense | Shared unchanged baseline |
| P1 | 25%, 50%, 75% | Per-output-row absolute weight ranking |
| P2 | 25%, 50%, 75% | Per-row Wanda ranking using calibration input-column L2 norms |
| P3 | 50%, 75% | Projection-specific Wanda quotas at the exact global P1/P2 zero count |
| Q1 | 8, 4, 3 bits | Uniform group-wise asymmetric PTQ |
| Q2 | 8, 4, 3 bits | Uniform AWQ-style channel scaling and the Q1 quantizer |
| Q3 | INT4 and INT3 byte budgets | Block-specific choices from 2, 3, 4 and 8 bits using Q2 |
| Wanda 2:4 | 50% | Two highest scores in every contiguous group of four input weights |

**Calibration and allocation.** Channel statistics use all 65,536 tokens in the 128 locked calibration sequences. Projection-output reconstruction probes use four fixed positions (31, 159, 287, 415, zero-based) in every sequence. Each probe divides the sum of squared output differences by dense output energy on those positions. The 512 probe tokens, frozen dense inputs, exact channel norms and sampled score quantiles make the local errors comparable. These isolated probes do not model interactions between compressed blocks.

- P1 and P2 use stable row-wise ordering. Original zero weights win score ties, then input-column order breaks ties. P3 measures sensitivity at 50% Wanda pruning. It assigns quotas proportional to inverse sensitivity raised to 0.25 or 0.5, clips each projection to 10–85%, and distributes integer remainders across projections and rows. Both candidates are evaluated on the separate 32 development windows. Lowest pooled development NLL wins, with the first strength winning a tie. The output retains both candidates.
- Q2 searches exponents 0, 0.5 and 1 on mean absolute calibration activations. It normalizes channel scales by the geometric mean of their extrema, stores inverse scales in FP16, and selects the smallest local relative reconstruction error. Before rounding weights it verifies the scaled and original FP32 projection outputs. O scales are shared over repeated query heads so their reciprocals fold into V. Down reciprocals fold into up. Folded producer biases keep FP16 precision. Independent Q/K/V and gate/up input scales use explicit FP16 vectors. The dense execution copy combines the explicit inverse with the dequantized weight. This reproduces AWQ-style channel scaling and a limited calibration search. Execution uses dense PyTorch kernels.
- Q3 measures each block’s summed relative projection error at all four widths. A multiple-choice knapsack minimizes total error under each full-model byte ceiling and requires a mixed assignment. It reuses Q2’s selected scales and quantizer for the assigned width. All widths, selection criteria, candidate scores, and rejected P3 candidates are frozen before held-out scoring.

**Storage.** All estimates use a known Qwen architecture and a fixed tensor order. For Q1/Q2/Q3, group 128 contiguous input columns per row, right-pad the last group with zeros, and include zero in each group’s representable range. The scale is the range divided by the largest code, rounded to FP16 with a floor of 2⁻²⁴. Round to nearest, ties to even. Store one unsigned byte zero-point per group and clip unsigned codes to the bit-width range. The specified payload packs row-major codes contiguously, least significant bits first, with each projection byte aligned. Charge code bytes, two bytes per group scale, one byte per zero-point, explicit FP16 inverse vectors, one width byte per block and all excluded FP16 parameters. The runner estimates this payload and serializes the explicit scale vectors as separate evidence. PyTorch archive overhead belongs to the evidence file format. The sparse estimate uses one bit per eligible weight (rounded up per matrix) and two bytes per retained nonzero. The 2:4 estimate uses a three-bit index for one of six pairs per quartet plus two FP16 values. All estimates are reported alongside dense FP16 storage and measured GPU allocation.

**Evaluation and timing.** Each model scores exactly 50,000 targets per corpus with the locked, independently evaluated WikiText-2 test and C4 validation windows. The first token in each window supplies context. Batched scoring right-pads with attention masks, groups similar lengths, and computes the vocabulary head in bounded chunks for scored positions only. HellaSwag uses the zero-shot harness v0.4.8 task version 1.0. The prompt joins the activity label, a colon and space, context A, a space, and capitalized context B. The harness cleanup processes title markers, bracketed text and extra spaces. Each candidate is preceded by a space. Normalization divides the conditional log likelihood by the number of characters in the cleaned ending. Candidate scores and both predictions retain the supplied ID order.

FP16, P1/P2/P3 at 50%, and Q1/Q2/Q3 at INT4 and INT3 score all 2,000 HellaSwag examples. Every other condition scores the first 512. Baseline deltas use identical IDs. The quality plots use the common 512-ID diagnostic subset. Timings measure dense FP16 execution on batch-one 512- and 2,048-token inputs, each followed by 64 cached greedy decode steps. Each shape uses two warm-ups and five CUDA-synchronized repetitions. Timing prompts come from fixed calibration IDs. Packed-storage estimates, actual dense-kernel latency and peak allocated CUDA bytes are recorded separately.

## Evidence

| Path under `results/task1/` | Contents |
| --- | --- |
| `protocol.json`, `environment.txt`, `inputs/` | Versions, checkpoint, seed, source and code hashes, exact windows, IDs, prompt text, tokenized alternatives |
| `calibration.json`, `plan.json` | Calibration effort, position selection, score distributions, local errors, scale search, development scores, selected allocations and byte checks |
| `<condition>/` | Window NLL sums and denominators, four scores per HellaSwag ID, raw and normalized accuracy, projection diagnostics, storage components, numerical settings, stage times, timing samples and memory |
| `summary.csv`, `status.json` | Numerical grid, matched baseline deltas, estimated storage ratios, measured timing ratios and completion state |
| `paired_intervals.json` | Adaptive-minus-uniform changes and paired 95% percentile intervals from 2,000 seed-6304 bootstrap draws. Resample whole windows and exponentiate pooled NLL for PPL, or resample matched HellaSwag IDs for accuracy |
| `quality.png`, `allocations.png`, `calibration_vs_heldout.json` | Separate pruning/quantization trade-offs, sensitivity and allocation maps, and paired calibration/held-out changes |
| `qualitative.json` | First four manifest examples plus the first newly correct and first newly incorrect normalized prediction per condition, when present. Includes input, alternatives, labels and aligned scores |
| `cache/` | Regenerable calibration tensors for resuming sessions |

The smoke suite runs random miniature Qwen weights on CPU, checks masks, quantization, exact budgets, likelihoods, restoration, bootstrap pairing and all 18 condition paths. Temporary results are deleted automatically. Real-model results are generated by the experiment command.

## Run Task 2 on Kaggle

Use a Kaggle notebook with Internet and two CUDA GPUs. Run this setup cell first in each notebook session. It reuses the assignment checkout at the path below. Task 2 reads the frozen MQAR and KV retrieval assets directly from the course checkout. The runner clones it when absent and checks the four required file hashes against the manifest in the pinned instructor commit. You can also reuse a checkout containing your locally prepared Task 1 files.

```python
import os
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/pa4")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/hyderchaudhry/atml-pa4.git", str(repo)], check=True)
os.chdir(repo)
if not Path("tests/test_task2.py").is_file():
    raise FileNotFoundError("Use an assignment checkout containing Task 2 and its smoke tests.")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_task2.py", "-v"], check=True)
```

The offline smoke tests use synthetic tensors, a tiny random Qwen model, and temporary files. To run the experiments, use one notebook cell per part. Each invocation launches one independent worker per GPU. The notebook waits for both workers and builds that part’s evidence before returning.

```python
!python -m task2.run --part a --gpus 2
```

```python
!python -m task2.run --part b --gpus 2
```

```python
!python -m task2.run --part c --gpus 2
```

For a single invocation, use `!python -m task2.run --part all --gpus 2`. Part B comprises **42 independent training runs and 189,000 optimizer updates**. Save the whole `results/task2/` directory as Kaggle output between sessions, copy it back to the writable clone, and repeat the same command. The runner skips completed shapes, training runs, retrieval cases and systems conditions. MQAR checkpoints include weights, optimizer, loss scaler and completed update number every 500 updates. Run one launcher per output directory.

| Part and output | Fixed experiment and saved evidence |
| --- | --- |
| `results/task2/a/` | Batch 2, four heads, dimension 64, six lengths from 256 through 8192, FP16 and FP32. Per-shape JSON, `operators.csv` and cost plots include output and Q/K/V gradient errors, causality and extreme-score probes, forward and full forward-plus-backward timings, peak allocated/reserved bytes and backend failures. Each supported timing uses two warmups and ten synchronized repetitions. |
| `results/task2/b/` | Dense, window, ELU+1 linear, mixed local/dilated, and a near-first window curriculum. Three seeds per training condition, 4,500 updates of 32 examples. Per-run checkpoints, training losses, six development evaluations, final query-level predictions and losses at each test load, and post-hoc window controls on dense checkpoints. `recall.csv`, `learning.png`, `distance_load.png`, generator checks and the independent length sweep are saved here. |
| `results/task2/c/` | All six policies on all 180 frozen Qwen2.5-1.5B-Instruct prompts. `matched_budget.csv`, `factorial.csv`, `paired_accuracy.csv`, `reference_checks.json` and `aligned_cases.json` cover fidelity and evidence selection. Per-case files retain actual lengths, prompts by course ID, decoded answers, packed byte components and memory/timing measurements. `systems/` and `systems.csv` cover actual pack, unpack and decode-step costs for batches 1 and 8 at lengths 2048 and 4096. |

**Part A implementation.** Both GPUs divide the length/dtype grid. Operators use identical seeded inputs and upstream gradients within a shape. The Python tiled implementation uses 128- and 256-position tiles, FP32 online-softmax accumulators and a recomputing backward pass. Every causal row includes its diagonal. The runner forces each named SDPA backend independently and records unsupported operations or allocation failures. Memory measurements include live inputs, reference outputs when feasible, and the PyTorch allocator.

**Part B implementation.** Each GPU trains a separate model with the full batch. The two pre-norm blocks use hidden size 128, four 32-dimensional heads, a 512-wide GELU MLP, learned token embeddings and sinusoidal positions. AdamW uses peak LR 0.0008, weight decay 0.01, 200 warmup updates, linear decay to 15% of peak and global norm clipping at 1.0. CUDA training uses FP16 autocast and dynamic loss scaling. Overflow retries reuse the same batch. Loss is over 64 possible values at the supplied query positions. Training-stream chunks retain source/query positions, answers, distances and input tokens. Matched runs reuse the instructor generator indices. Final evaluation uses 1,024 frozen sequences in each load/regime cell and also evaluates dense checkpoints with the window imposed at inference.

Window and dilated heads use dense boolean masks. A two-layer 64-position window reaches at most 126 positions. Linear attention uses FP32 recurrent state plus causal products within 32-token chunks. The forward-only operator sweep uses matched random Q/K/V with batch 2, four heads and dimension 32. It reports theoretical pairs and state bytes separately from measured CUDA time and memory. The intervention uses the window architecture at K=16 with a near-only prefix followed by mixed training. Three mixed-window development scores determine the prefix length under the recorded rule. Each intervention directory contains its prediction and selection inputs before training starts. Review the dense baseline learning curves before interpreting other models.

**Part C implementation.** Each GPU owns complete retrieval prompts and runs all policies on each assigned prompt. The model uses the pinned Instruct revision, FP16 and SDPA. The runner splits the tokenized chat at the final `QUESTION_BEGIN` marker and scores the pre-question cache using rotary-positioned suffix queries, causal FP32 attention, and a mean across query positions, heads and layers. Selection uses one ordered position set shared across layers and heads. Positions keep their original RoPE coordinates; physical append positions follow the compact cache length. The answer extractor takes the first standalone five-digit code from up to 16 greedily generated tokens. Each case records the direct versus split reference check. Inspect failed reference checks before interpreting compressed accuracy.

The budget is the actual INT4 channel-wise-K/token-wise-V prefix payload. Groups have 64 elements, one FP16 scale and one FP16 minimum offset. Unsigned codes use nearest-even rounding and low-bit-first byte packing. Each incomplete group stays in FP16. Both eviction policies and the hybrid protect eight sink and 64 recent tokens and charge four bytes per separately stored selected index. The hybrid charges INT2 data, metadata, selected FP16 values and remaining FP16 residuals. The implementation dequantizes for standard SDPA and re-quantizes newly complete 64-position groups during generation. Results distinguish serialized tensor bytes, appended tokens at FP16 cost, live packed-equivalent bytes, dense execution cache bytes and peak CUDA allocation. Per-prompt decode timings include a separate suffix and greedy breakdown. The independent systems sweep uses deterministic synthetic KV tensors at the model’s exact dimensions and ten warmed, synchronized repetitions, including an FP16 reference with matched physical shape for each decode comparison. Bootstrap intervals use 2,000 paired prompt resamples over the full evaluation set. The first ID in each factorial cell supplies the fixed aligned examples.

The operator designs follow the manual’s FlashAttention, Longformer, linear-attention and MQAR readings. Question-aware retention and asymmetric K/V grouping follow the SnapKV and KIVI motivations. Data and generator code come from the pinned instructor repository. [Course data repository](https://github.com/AbDu11aHHH/atml-pa4-datasets)

## Sources and attribution

- Data selection, tokenization windows, materialization and checksums use the course repository at commit `6a0a9ec9da7ef3d313ee3036cfec493886723644`. [Course data repository](https://github.com/AbDu11aHHH/atml-pa4-datasets)
- Prompt cleanup, conditional choice formatting and character normalization follow EleutherAI’s HellaSwag implementation. [Harness task, v0.4.8](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8/lm_eval/tasks/hellaswag)
- The pruning score and channel-rescaling methods follow the Wanda and AWQ papers. The projection-error sensitivity and constrained allocation rules are implemented here, informed by reconstruction-based post-training quantization. [Wanda](https://arxiv.org/abs/2306.11695) · [AWQ](https://arxiv.org/abs/2306.00978) · [BRECQ](https://arxiv.org/abs/2102.05426)
- Model inference uses PyTorch and Hugging Face Transformers. Codex assisted with implementation and smoke tests. [Qwen2.5-1.5B base model](https://huggingface.co/Qwen/Qwen2.5-1.5B)

## Run Task 3 on Kaggle

Start a fresh notebook with **Internet enabled** and the **GPU T4 ×2** accelerator. Use a revision of your assignment repository that contains `task3/run.py` and `tests/test_task3.py`. Both cells below use the notebook’s Python executable.

**Cell 1 · Clone, install, and smoke-test**

```python
import os
import subprocess
import sys
from pathlib import Path

repo = Path("/kaggle/working/pa4")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/hyderchaudhry/atml-pa4.git", str(repo)], check=True)
os.chdir(repo)
if not Path("task3/run.py").is_file():
    raise FileNotFoundError("This checkout needs the revision containing Task 3.")

def run(*args):
    subprocess.run([sys.executable, *args], check=True)

run("-m", "pip", "install", "-r", "requirements.txt")
run("-c", "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2'")
run("-m", "unittest", "discover", "-s", "tests", "-p", "test_task3.py", "-v")
```

Smoke tests use miniature models and temporary directories. They exercise the production sampler, loss and checkpoint path, paired timings, and evidence aggregation. The two-device test runs when both GPUs are visible. You can stop after this cell.

**Cell 2 · Run the full study**

```python
run("-m", "task3.run", "--gpus", "2")
```

To run in stages, replace cell 2 with the same command plus, for example, `"--stage", "train"`. The stages in order are `validate`, `baseline`, `pool`, `train`, `banks`, `alignment`, `speed`, `quality`, and `report`. Each invocation completes outstanding prerequisites. A successful full run writes `results/task3/stages/report.json`. Save the **whole** `/kaggle/working/pa4/results/task3/` as Kaggle output. In a new session, restore that folder to the writable clone before rerunning cell 2. Keep the code, package versions and GPU configuration consistent. Training checkpoints include the adapter, optimizer, loss scaler, prompt cursors and token counts every eight completed updates and at the final update. Completed prompt files are reused. Allow disk space for four merged FP16 drafts (about 4 GB together), the original Hugging Face model cache, and raw evidence.

**Data in a fresh clone.** Task 3 fetches the pinned instructor checkout into `datasets/` when absent. It verifies the short-form examples and GSM8K ID manifests against the instructor commit. It reuses prepared GSM8K files with the matching source-revision lock or loads the prescribed rows from the pinned GSM8K source. The assignment repository can therefore omit the instructor checkout and its generated binaries. The public-data setup in the Task 1 section also prepares the locked GSM8K rows. The runner uses complete instructor-generated prefix banks when present at `datasets/data/prefix_banks/prefix_banks_locked.jsonl.gz` alongside the matching `datasets/data/local_source_revisions.json`. Otherwise it calls the pinned instructor’s `setup_data.py` prefix-bank routine, with a compatibility adapter for its per-prompt random generator on Transformers 4.57.1. It saves the generated bank under `results/task3/banks/course/`. The exact-speculative control uses the same checkpoint generation settings as those course banks, including their repetition penalty.

**How both GPUs are used.** Each GPU holds its own target and draft for timing. Both GPUs score different held-out prompts, with every draft and its adjacent target-only reference for a given prompt on the same GPU. During training, each worker owns a complete method/seed run. GPU 0 trains off-policy seed 6304 and on-policy seed 6305. GPU 1 trains the complementary pair. This preserves microbatch 1 and the global 256-scored-token update. Rollout preparation and standalone accuracy use batched generation. Timed requests run individually. The course bank materializer runs on one device while the other starts its own rollout banks. The final aggregation runs on CPU.

| Evidence under `results/task3/` | Contents |
| --- | --- |
| `protocol.json`, `inputs/`, `checks/` | Revisions, environment, schedule, tokenized prompts and IDs, the 100,000-sequence Markov test and negative control, forced EOS/bonus/rollback cases, and real-model cache equivalence on each device. |
| `teacher_pool/`, `students/`, `training.csv`, `loss.png` | Frozen teacher rollouts, four independent training histories, rollout seeds and termination, scored prefixes, overflow counts, resource costs, FP32 adapters, merged FP16 drafts, and logit-error checks. Each run scores 30,000 math and 10,000 short-form tokens in 156 full updates and a final 64-token update. |
| `banks/`, `prefix_metrics/`, `alignment.csv` | The three common banks and four trained-student banks. All five drafts score the same saved prefix positions. Includes both KL directions, TV, expected acceptance and greedy agreement by workload and position. `target_spec_consistency.json` and `disagreements.json` contain paired controls and aligned examples. |
| `speed/`, `costs/`, `per_prompt_speed.csv`, `speed_summary.csv` | Two paired repetitions at proposal lengths 2 and 4, a 64-token EOS-aware cap on all 120 prompts, and a fixed-64 diagnostic on the first 24 math prompts. Includes an unmerged adapter timing control, warmup, prefill, four synchronized round phases, host overhead, actual model calls, peak allocated bytes, measured and predicted speedup, and break-even acceptance. |
| `quality/`, `standalone_accuracy.csv`, `standalone_contrasts.json` | Two standalone draft samples per prompt. Math uses a 512-token cap and the last numeric answer after the final `####` marker, falling back to the last number when the marker is absent. Short form uses 32 tokens, exact stripped answers and family-specific format checks. Outputs retain text, extracted answer, EOS and truncation. |
| `paired_method_differences.csv`, `seed_dispersion.csv`, `aligned_continuations.json` | Paired prompt bootstrap intervals with 2,000 resamples, variation across training seeds, and matched-prompt generations. |

The primary speedup is the per-prompt speculative tokens/second divided by the adjacent target-only tokens/second, including prefill and timing instrumentation. Isolated cost samples use warmed caches at 32, 128 and 512 tokens. The target’s verification forward consumes the pending token and the proposals. The cost projection uses the closest measured prefix length. The amortization calculation charges each off-policy deployment the full teacher-pool preparation cost and reports the request count only when measured seconds saved per request are positive.

The exact-sampling implementation follows Leviathan et al. The two distillation conditions follow the manual’s DistillSpec and on-policy distillation readings. The chat formatting and model/KV-cache APIs come from Hugging Face Transformers. The course repository supplies the data manifests, prefix-bank generation routine and position-selection rule. [Course data and generator](https://github.com/AbDu11aHHH/atml-pa4-datasets)
