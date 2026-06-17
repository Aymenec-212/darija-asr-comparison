# Reproducing the experiments

This document is everything you need to replicate the Phase 1 results from a clean machine.

---

## Hardware

The full Phase 1 (4 fine-tuning runs + zero-shot baselines + post-training evaluation) requires:

- **1× GPU with ≥ 40 GB VRAM** (A40, A100, L40, or RTX 6000 Ada all fit comfortably)
- **100 GB disk** (model weights ~22 GB, MoulSot audio cache ~25 GB, checkpoints/results ~15 GB, headroom ~38 GB)
- **~18 GPU-hours total** (see breakdown below)

We used [RunPod](https://www.runpod.io/) community-cloud A40 instances at ~$0.35-$0.50/hr, putting the total compute cost at **~$6-$9**.

### Why these specs

| Constraint            | Driver                                                                                       |
|-----------------------|----------------------------------------------------------------------------------------------|
| ≥ 40 GB VRAM          | Qwen2-Audio-7B in bf16 + activations + LoRA optimizer state ≈ 32 GB at per_device_batch=4    |
| 100 GB disk           | Two base models + audio cache + per-run trainer checkpoints (2.5 GB × 3 each, rotated)       |
| Network volume        | Survives pod restarts; checkpoint mid-run if you need to swap to a cheaper GPU for evaluation |

---

## Software

### Pod template (RunPod)

```
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

This image preinstalls PyTorch 2.4.0 + CUDA 12.4 + Python 3.11. **Do NOT reinstall PyTorch** — the notebooks' pip install cells use `>=` specifiers that will not downgrade.

Mount your **persistent network volume** at `/workspace`. The notebooks read/write under `/workspace/outputs/`.

### Dependencies

Install with:

```bash
pip install -r requirements.txt
apt-get install -y ffmpeg   # required by torchcodec for audio decoding
```

Or let the notebooks do it — Cell 3 of each notebook contains the same pinned `pip install` block, and Cell 5 imports + smoke-tests `torchcodec`.

### Hugging Face Hub access

You'll need a write-scope token to push the fine-tuned adapters:

1. Create token at https://huggingface.co/settings/tokens (role = **Write**).
2. Set it as environment variable:
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxx"
   ```

(Read-only access is sufficient if you only want to *use* the pre-trained adapters, not push your own.)

---

## Datasets

Both load automatically the first time the notebooks run, via `datasets.load_dataset()`:

| Dataset                | Used for                                        | Size on disk |
|------------------------|-------------------------------------------------|--------------|
| `atlasia/MoulSot-Full` (config `100-gt-2.5`) | Training + in-domain test  | ~22 GB       |
| `UBC-NLP/Casablanca` (Morocco subset, test split) | OOD evaluation             | ~3 GB        |

**Caveat:** Hugging Face's dataset cache lives at `~/.cache/huggingface/datasets/`. If your pod's home directory is *not* on the persistent network volume, this gets wiped on pod restart and re-downloads on next run. To avoid this:

```bash
export HF_DATASETS_CACHE="/workspace/.cache/datasets"
```

before launching Jupyter.

---

## Running the experiments

### Order of operations

The notebooks are designed to run **independently** but for the cleanest comparison run them in this order:

1. **`notebooks/whisper_v3_darija_finetune.ipynb`** with `DATA_SCALE_HOURS = 10`
2. Same notebook, change Cell 1 to `DATA_SCALE_HOURS = 30`, restart kernel, re-run
3. **`notebooks/qwen2audio_darija_finetune.ipynb`** with `DATA_SCALE_HOURS = 10`
4. Same notebook, change Cell 1 to `DATA_SCALE_HOURS = 30`, restart kernel, re-run

Reason for order: the Whisper runs are faster (3.4 hr at 30h vs 8.9 hr for Qwen2-Audio), so you discover any environment issues early. Qwen2-Audio's longer 30h run is then the last thing to launch.

### Per-run wall-clock estimates (NVIDIA A40)

| Run                   | Wall-clock | What it produces                                                              |
|-----------------------|-----------|-------------------------------------------------------------------------------|
| Whisper-LV3 zero-shot | ~30 min   | `zero_shot_results_whisper_lv3.json` (test) + Casablanca OOD                  |
| Whisper-LV3 10h FT    | 2.2 hr    | `best_marco_asr_whisper_10h/`, `final_test_whisper_10h_*.json`, history, OOD  |
| Whisper-LV3 30h FT    | 3.4 hr    | Same as above with `_30h` suffix                                              |
| Qwen2-Audio zero-shot | ~1 hr     | Both prompt variants (strict + Darija)                                        |
| Qwen2-Audio 10h FT    | 2.1 hr    | `best_marco_asr_10h/`, `final_results_10h.json`, history, OOD                 |
| Qwen2-Audio 30h FT    | 8.9 hr    | Same as above with `_30h` suffix                                              |

### Restart vs resume

If a run gets killed (disk fill, pod restart, OOM), **re-running the same notebook auto-resumes** from the latest valid checkpoint. The Marco-ASR callback replays its history JSON to restore `best_wer` and `no_improve_count` so early-stopping picks up exactly where it left off.

For a **clean fresh run** (e.g. you changed a hyperparameter), delete the **three coupled artifacts** for that scale:

```bash
rm -rf /workspace/outputs/trainer_10h/
rm -rf /workspace/outputs/best_marco_asr_10h/
rm    /workspace/outputs/logs/marco_history_10h.json
```

Deleting only the trainer dir while leaving the history JSON will cause stale-state leakage that can premature-stop the fresh run.

---

## After all four runs complete

Once you have all four `final_test_*_summary.json` / `final_results_*.json` files, run the aggregation script to regenerate the CSVs in `results/`:

```bash
# (planned - aggregation script not yet checked in)
python scripts/aggregate_results.py /workspace/outputs/results /workspace/outputs/logs --out results/
```

The aggregation produces:

- `aggregate_metrics.csv` — every (model × scale × dataset × partition) row
- `training_costs.csv` — wall-clock, best-step, convergence
- `learning_curve_points.csv` — WER vs hours
- `generalization_and_codeswitch_gaps.csv` — gaps
- `learning_efficiency.csv` — pp-per-train-hour
- `lora_footprint.csv` — module counts

These are the inputs to the diagnostic plots (in `results/plots/`).

---

## Paired-bootstrap comparison (Qwen2-Audio vs Whisper at each scale)

The headline finding ("Whisper beats Qwen2-Audio") rests on paired-bootstrap delta-WER intervals computed across the **same 1962 test utterances**. The notebooks save per-utterance `*_details.json` files; the paired bootstrap consumes both files (one per architecture) and resamples utterance indices 1000× to derive a 95% CI on the WER delta.

To run the comparison after both FT runs at a given scale (e.g. 10h):

```python
# In a fresh notebook cell:
import json, numpy as np
from jiwer import wer as jiwer_wer

with open("/workspace/outputs/results/final_test_whisper_10h_details.json") as f:
    whisper = json.load(f)
with open("/workspace/outputs/results/final_results_10h.json") as f:
    qwen = json.load(f)["moulsot_test"]["detail_compact"]["full"]

# Match by reference (both notebooks process the test set in the same order)
n = min(len(whisper), len(qwen))
refs = [whisper[i]["norm_ref"] for i in range(n)]
ft_whisper_preds = [whisper[i]["norm_pred"] for i in range(n)]
ft_qwen_preds = [qwen[i]["norm_pred"] for i in range(n)]

# Delta = WER(qwen) - WER(whisper); positive = whisper wins
rng = np.random.RandomState(42)
deltas = []
for _ in range(1000):
    idx = rng.choice(n, size=n, replace=True)
    r = [refs[i] for i in idx]
    w = [ft_whisper_preds[i] for i in idx]
    q = [ft_qwen_preds[i] for i in idx]
    deltas.append(float(jiwer_wer(r, q)) - float(jiwer_wer(r, w)))

print(f"Δ WER (Qwen2-Audio - Whisper) at 10h:")
print(f"  point:   {float(jiwer_wer(refs, ft_qwen_preds)) - float(jiwer_wer(refs, ft_whisper_preds)):.4f}")
print(f"  95% CI:  [{np.percentile(deltas, 2.5):.4f}, {np.percentile(deltas, 97.5):.4f}]")
```

If the CI excludes 0 (and both CI bounds are positive), Whisper's improvement is statistically significant at α=0.05.

---

## Troubleshooting

The Phase 1 notebooks include hardening for the failures we hit during development. The most common ones:

| Symptom                                                                      | Cause                                                                                      | Fix                                                                                                              |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `ImportError: To support decoding audio data, please install 'torchcodec'`   | `datasets>=3.0` requires torchcodec for Audio columns                                      | `apt-get install -y ffmpeg && pip install torchcodec` (already in `requirements.txt`)                            |
| `TypeError: WhisperForConditionalGeneration.forward() got 'input_ids'`       | `task_type=SEQ_2_SEQ_LM` in LoRA config wraps Whisper with the wrong PEFT class            | Omit `task_type` in `LoraConfig` (base `PeftModel` forwards kwargs as-is)                                        |
| `UnpicklingError: ... weights_only=True ...`                                 | PyTorch 2.6+ stricter `torch.load` default                                                 | The notebook patches `torch.load` to default `weights_only=False` in Cell 5; restart kernel                      |
| `RuntimeError: ... 8 GB free`                                                | `DiskSpaceGuardCallback` aborted before a save would fail                                  | Free disk on `/workspace` and resume                                                                              |
| `IsADirectoryError: ... .ipynb_checkpoints`                                  | Jupyter dropped a checkpoint dir inside an adapter folder                                  | `rm -rf <dir>/.ipynb_checkpoints`; the push scripts now skip directories+hidden files                            |
| Marco-ASR prematurely early-stops at step 300 of a fresh run                 | Stale `marco_history_*.json` from a previous run is being replayed by the callback         | Delete the trio: `trainer_*/`, `best_marco_asr_*/`, `marco_history_*.json` before launching                       |
| Eval batch=8 OOMs on Whisper-LV3 with 30s audio                              | Encoder activations × max time dim are the bottleneck                                      | Stay at `MARCO_EVAL_BATCH_SIZE=4` (already the notebook default)                                                  |

---

## Verifying you got the same numbers

If your reproduction WERs differ from the headline results by more than the bootstrap CIs in [`docs/RESULTS.md`](RESULTS.md), check:

1. **Seed**: `SEED=42` in Cell 1, and you did NOT change it.
2. **Filter logic**: `MAX_AUDIO_DURATION_SEC = 30.0`. If you raised this, your filtered counts will differ (test set should be 1962 after filtering, not 1993).
3. **Casablanca normalization**: did not get edited. The normalization function is byte-identical between the two notebooks.
4. **PEFT version**: `peft >= 0.11.0`. The LoRA-with-`all-linear` target selection logic in older PEFT versions sometimes missed modules.
5. **Eval cadence at your scale**: `MARCO_EVAL_SUBSET_SIZE` and `MARCO_EVAL_STEPS` are scale-keyed dicts that auto-update when you change `DATA_SCALE_HOURS`. Verify with `print(MARCO_EVAL_SUBSET_SIZE)`.
6. **Effective batch size**: per-device 4 × grad-accum 4 = 16. If your GPU has less VRAM and you halved per-device batch, double grad-accum to preserve the effective batch.
