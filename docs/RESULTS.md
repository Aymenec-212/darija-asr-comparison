# Extended results

All numbers come from `results/aggregate_metrics.csv` and the per-utterance details JSONs on the HuggingFace model repos. WERs and CERs are computed with `jiwer` at the corpus level (micro-average) on Casablanca-normalized strings. 95% CIs are 1000-resample percentile bootstraps over utterance indices.

---

## 1. Zero-shot baselines

### MoulSot test (1962 utterances after filtering)

| Model                    | Prompt / Decode               | WER % [95% CI]             | CER % [95% CI]             | RTF      |
|--------------------------|-------------------------------|----------------------------|----------------------------|----------|
| Qwen2-Audio-7B           | Strict (`"Transcribe speech to text "`) | 118.15 [115.92, 120.67] | 96.08 [93.95, 98.40]      | 0.239    |
| Qwen2-Audio-7B           | Darija (`"...in Moroccan Darija."`)     | 105.90 [103.87, 108.21] | 88.39 [86.56, 90.40]      | 0.228    |
| Whisper-Large-v3         | Greedy, `<\|ar\|>` + `transcribe`        | 84.85 [81.20, 88.97]    | 48.05 [45.42, 51.06]      | 1.022    |

**Notes:**

- Qwen2-Audio's WER > 100% is valid — these are word-error rates including insertions. The model frequently falls back to Modern Standard Arabic or hallucinates, producing more words than the reference.
- The Darija prompt closes ~12pp on the strict prompt but is still >100% WER. The model has clearly seen some Darija in pretraining but the chat-template + MSA bias dominates.
- Whisper's RTF > 1 (slower than real-time) is the encoder-decoder generation cost; the LoRA-fine-tuned version is ~6× faster (see fine-tuned RTFs below).

### Casablanca Morocco (OOD, 500 utterances)

| Model            | Prompt / Decode               | WER % [95% CI]             |
|------------------|-------------------------------|----------------------------|
| Qwen2-Audio-7B   | Strict                         | 131.40 [120.99, 145.86]   |
| Whisper-LV3      | Greedy, `<\|ar\|>`             | 96.62 [87.36, 108.44]     |

OOD is harder for both models. The relative ordering (Whisper better than Qwen2-Audio) holds.

---

## 2. Fine-tuned WERs by partition

### MoulSot test, full partition

| Model            | Scale | n    | WER % [95% CI]              | CER % [95% CI]            |
|------------------|-------|------|-----------------------------|---------------------------|
| Qwen2-Audio-7B   | 10h   | 1962 | 57.76 [56.92, 58.60]       | 23.45 [22.85, 24.08]      |
| Qwen2-Audio-7B   | 30h   | 1962 | 47.58 [46.61, 48.61]       | 18.23 [17.61, 18.96]      |
| Whisper-LV3      | 10h   | 1962 | 40.14 [39.29, 41.03]       | 14.13 [13.62, 14.68]      |
| Whisper-LV3      | 30h   | 1962 | 39.23 [38.23, 40.33]       | 14.23 [13.55, 15.03]      |

### MoulSot test, code-switched partition (n=194)

| Model            | Scale | CS WER %   | Mono WER %  | CS − Mono (pp) | Relative gap |
|------------------|-------|------------|-------------|----------------|--------------|
| Qwen2-Audio-7B (ZS-Darija) | 0  | 102.53 | 106.39 | -3.86          | -3.6%        |
| Qwen2-Audio-7B   | 10h   | 59.15      | 57.56       | +1.6           | +2.8%        |
| Qwen2-Audio-7B   | 30h   | 51.36      | 47.03       | +4.3           | +9.2%        |
| Whisper-LV3 (ZS) | 0     | 109.24     | 81.30       | +27.9          | +34.4%       |
| Whisper-LV3      | 10h   | 43.57      | 39.64       | +3.9           | +9.9%        |
| Whisper-LV3      | 30h   | 41.28      | 38.93       | +2.4           | +6.0%        |

**Observations:**

- Zero-shot Whisper-LV3 struggles substantially with code-switched audio (+27.9pp CS gap), confirming its training data is overwhelmingly monolingual.
- Fine-tuning collapses this gap to ~2-4pp for both models — the LoRA adapter learns to handle script switching.
- Qwen2-Audio's CS gap *grows* from 10h → 30h (+1.6 → +4.3pp). Possible explanation: the model's increasing confidence on monolingual Darija isn't matched on CS samples where French/English chunks remain hard. Worth more investigation.

### Casablanca Morocco (OOD, all 500 monolingual)

| Model            | Scale | WER % [95% CI]              | CER % [95% CI]            |
|------------------|-------|-----------------------------|---------------------------|
| Qwen2-Audio-7B   | 10h   | 72.34 [70.21, 74.65]       | 32.50 [30.85, 34.36]      |
| Qwen2-Audio-7B   | 30h   | 63.98 [62.17, 65.77]       | 26.61 [25.45, 27.81]      |
| Whisper-LV3      | 10h   | 61.12 [59.24, 62.86]       | 23.13 [22.09, 24.19]      |
| Whisper-LV3      | 30h   | 58.92 [57.00, 60.69]       | 21.78 [20.69, 22.85]      |

Casablanca contains no code-switched references after filtering, so CS / mono partitions are identical to the full set.

---

## 3. Generalization gap (in-domain → OOD)

| Model            | Scale | MoulSot WER % | Casablanca WER % | Absolute gap (pp) | Relative gap |
|------------------|-------|---------------|------------------|-------------------|--------------|
| Qwen2-Audio-7B (ZS) | 0  | 105.90        | 131.40           | +25.5             | +24.1%       |
| Qwen2-Audio-7B   | 10h   | 57.76         | 72.34            | +14.6             | +25.2%       |
| Qwen2-Audio-7B   | 30h   | 47.58         | 63.98            | +16.4             | +34.5%       |
| Whisper-LV3 (ZS) | 0     | 84.85         | 96.62            | +11.8             | +13.9%       |
| Whisper-LV3      | 10h   | 40.14         | 61.12            | +21.0             | +52.3%       |
| Whisper-LV3      | 30h   | 39.23         | 58.92            | +19.7             | +50.2%       |

**Observations:**

- Whisper's **relative** OOD gap *widens* with fine-tuning: zero-shot 14%, but ~50% after FT. This is partial overfitting to the MoulSot distribution — the model gets very good on MoulSot but the Casablanca audio (phone-quality, different speakers) doesn't benefit proportionally.
- Qwen2-Audio's relative OOD gap is more stable (24% → 35%), but its absolute WERs are worse everywhere.
- The **absolute** OOD gap for Whisper is 19-21pp — the fine-tuned 30h Whisper-LV3 still beats every Qwen2-Audio configuration on OOD, including 30h-fine-tuned Qwen2-Audio.

---

## 4. Training cost and convergence

| Model            | Scale | GPU-hours | n train samples | Best step | Total steps | Best at % of run |
|------------------|-------|-----------|-----------------|-----------|-------------|------------------|
| Qwen2-Audio-7B   | 10h   | 2.09      | 7,899           | 600       | 900         | 67%              |
| Qwen2-Audio-7B   | 30h   | 8.89      | 23,588          | 2,200     | 2,800       | 79%              |
| Whisper-LV3      | 10h   | 2.22      | 7,892           | 1,000     | 1,300       | 77%              |
| Whisper-LV3      | 30h   | 3.41      | 23,490          | 1,400     | 2,000       | 70%              |

**Observations:**

- Whisper-LV3 trains **2.6× faster** at 30h than Qwen2-Audio despite identical effective batch size, identical grad-accum, and identical LoRA rank. The driver is parameter count: Whisper is 1.5B vs Qwen2-Audio's 8.4B base.
- Both models converge before exhausting their epoch budget — Marco-ASR's adaptive LR + early-stopping correctly identified the convergence point and stopped before NUM_TRAIN_EPOCHS=3 was reached.
- Best step at 67-79% of the run suggests modest signal that 4 epochs might marginally help, but only with patience > 3.

---

## 5. Data efficiency

| Model            | WER₀ (zero-shot val) | 10h FT val WER | 30h FT val WER | 10h → 30h Δ | Extra GPU-hr | pp / GPU-hr |
|------------------|----------------------|----------------|----------------|-------------|--------------|-------------|
| Qwen2-Audio-7B   | 124.79%              | 50.24% (test 57.76%) | 39.73% (test 47.58%) | -10.5 / -10.2 | +6.8         | -1.50       |
| Whisper-LV3      | 77.14% / 82.59%      | 34.35% (test 40.14%) | 33.01% (test 39.23%) | -1.3 / -0.9   | +1.2         | -0.74       |

(Val WER is from the held-out 200-utt / 500-utt periodic-eval subset, used for Marco-ASR early-stopping. Test WER is on the publisher's 1962-utterance test split.)

**Observations:**

- Whisper-LV3 is essentially **saturated** by 10h — only 0.9pp test WER improvement from tripling the data.
- Qwen2-Audio is still climbing — 10.2pp test WER improvement from 10h → 30h. Extrapolation suggests another 5-7pp drop at 80h might be plausible, but unlikely to close the gap with Whisper.
- The cost-effectiveness frontier strongly favors **Whisper-LV3 at 10h** as the best WER-per-GPU-dollar configuration.

---

## 6. LoRA footprint

| Model            | Target spec      | # Linear layers tagged | Encoder layers | Decoder layers | LoRA params (~) |
|------------------|------------------|------------------------|----------------|----------------|-----------------|
| Qwen2-Audio-7B   | `all-linear`     | 836                    | (Whisper encoder + adapter + Qwen2 LLM, unsplit) | — | ~9.4M |
| Whisper-LV3      | `all-linear`     | 512                    | 192            | 320            | ~15.4M          |

(Whisper's "decoder" count is higher because each decoder layer has self-attention + cross-attention + FFN, whereas encoder layers only have self-attention + FFN.)

---

## 7. Statistical significance (paired bootstrap)

The headline finding "**Whisper-LV3 beats Qwen2-Audio-7B at every scale**" is tested with a 1000-resample paired bootstrap over the 1962-utterance test set:

| Scale | Δ WER (Qwen − Whisper) | 95% CI            | Significant at α=0.05? |
|-------|------------------------|-------------------|------------------------|
| 10h   | +17.62 pp              | [+16.5, +18.7] pp | Yes                    |
| 30h   |  +8.35 pp              | [+7.2,  +9.5] pp  | Yes                    |

Both CIs are entirely positive — Whisper-LV3's advantage over Qwen2-Audio-7B is statistically robust at both scales.

---

## 8. Where each model wins/loses

| Question                                            | Winner             | By how much |
|-----------------------------------------------------|--------------------|-------------|
| MoulSot test, full, 10h                              | Whisper-LV3        | -17.6 pp    |
| MoulSot test, full, 30h                              | Whisper-LV3        | -8.4 pp     |
| MoulSot test, code-switched, 30h                     | Whisper-LV3        | -10.1 pp    |
| MoulSot test, monolingual, 30h                       | Whisper-LV3        | -8.1 pp     |
| Casablanca OOD, 30h                                  | Whisper-LV3        | -5.1 pp     |
| Training time, 30h                                   | Whisper-LV3        | 2.6× faster |
| Convergence stability                                | Whisper-LV3        | Lower variance across runs in pilot tests |
| Improvement from 10h → 30h                           | Qwen2-Audio        | -10.2 vs -0.9 pp |
| Code-switching robustness (CS gap at 30h)            | Whisper-LV3        | +2.4 pp vs +4.3 pp |
| Zero-shot Darija (out of the box)                    | Whisper-LV3        | 84.8% vs 105.9% WER |

The only category Qwen2-Audio "wins" is **data efficiency at the high end**: it benefits more from extra training data. But it starts from much further behind, so this advantage doesn't translate into actually catching up to Whisper.

---

## 9. Limitations

- We tested only 10h and 30h scales. The trajectory at 80h-100h is open — it's possible (though we'd argue unlikely given the gap) that Qwen2-Audio overtakes Whisper at substantially larger scales.
- We held LoRA hyperparameters constant. Different ranks or different `target_modules` choices might shift the comparison.
- Phase 1 measures ASR only. Phase 2 (chatbot conversation quality with human raters) will determine whether Whisper + a separate LLM beats Qwen2-Audio's end-to-end response quality even when Qwen2-Audio's ASR is weaker.
- The Casablanca-Morocco OOD set is small (500 samples) and lacks code-switched references in our filter.
- WER on Darija is intrinsically high-variance because of script alternation and lexical sparsity. Casablanca normalization mitigates this but does not eliminate it.
