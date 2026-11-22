# ATML PA4 · Task 1

Run the model-compression study on one CUDA GPU. The runner covers 18 conditions and writes evidence to `results/task1/`. Shared setup and experiment bookkeeping live in `common/`. The other task directories under `results/` are reserved for later work.

## Run on Kaggle

1. Enable **Internet** and a **GPU** in the notebook settings. Use the course environment lock if you have it. This checkout includes the course data repository’s dependency constraints. A separate full course environment lock was unavailable. The runner keeps Kaggle’s CUDA-enabled PyTorch, enforces Transformers 4.57.1 and huggingface_hub 0.36.2, and records the installed environment.
2. In a fresh Kaggle notebook, run the cells below. Replace the URL in the first cell with your repository’s clone URL.
3. Save `/kaggle/working/pa4/results/task1/` as Kaggle output. To continue a saved run in a new session, copy that directory, including its `cache/` subfolder and partial condition files, back to the same writable location after cloning. Use the same code, data, GPU type, and package versions for matched runs.

**Cell 1 · Clone and enter your repository**

```python
from pathlib import Path
import os
import subprocess
import sys

repo = Path("/kaggle/working/pa4")
if not repo.exists():
    subprocess.run(["git", "clone", "YOUR_REPOSITORY_URL", str(repo)], check=True)
os.chdir(repo)

def run(*args):
    subprocess.run([sys.executable, *args], check=True)
```

**Cell 2 · Install dependencies and prepare the locked data**

```python
run("-m", "pip", "install", "-r", "requirements.txt")
run("-m", "common.setup")
```

**Cell 3 · Smoke-test the implementation**

```python
run("-m", "unittest", "discover", "-s", "tests", "-v")
```

**Cell 4 · Run the full study when ready**

```python
run("-m", "task1.run")
```

C4 streaming can take time. To reuse the prepared files already present in this workspace, attach `datasets/data/public_locked/` as a Kaggle input and use `python -m common.setup --public-assets /kaggle/input/YOUR-DATA/public_locked`. Setup checks the five Task 1 asset hashes before reuse. The course script prepares any remaining public assets it requires. Model and tokenizer downloads use Hugging Face’s cache and the exact base-model revision in [the source lock](configs/source_revisions.json).

For a session split, freeze the plan first and then run one or more conditions. The runner includes the FP16 reference automatically. A later unrestricted invocation fills in the remaining conditions.

```bash
python -m task1.run --prepare-only
python -m task1.run --conditions p2_50 p3_50 q2_4 q3_4
python -m task1.run
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

## Sources and attribution

- Data selection, tokenization windows, materialization and checksums use the course repository at commit `6a0a9ec9da7ef3d313ee3036cfec493886723644`. [Course data repository](https://github.com/AbDu11aHHH/atml-pa4-datasets)
- Prompt cleanup, conditional choice formatting and character normalization follow EleutherAI’s HellaSwag implementation. [Harness task, v0.4.8](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8/lm_eval/tasks/hellaswag)
- The pruning score and channel-rescaling methods follow the Wanda and AWQ papers. The projection-error sensitivity and constrained allocation rules are implemented here, informed by reconstruction-based post-training quantization. [Wanda](https://arxiv.org/abs/2306.11695) · [AWQ](https://arxiv.org/abs/2306.00978) · [BRECQ](https://arxiv.org/abs/2102.05426)
- Model inference uses PyTorch and Hugging Face Transformers. Codex assisted with implementation and smoke tests. [Qwen2.5-1.5B base model](https://huggingface.co/Qwen/Qwen2.5-1.5B)
