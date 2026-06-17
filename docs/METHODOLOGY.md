# Methodology

This file is a **placeholder pointing to** the full methodology document.

The full Phase 1 methodology (~6,000 words, with research questions, hypotheses, design decisions, threats to validity, and protocol specifications) lives at:

> `Phase_1_Methodology__Comparing_End-to-End_Audio-LLMs_and_Cascaded_ASR_for_Darija_Transcription.md`

Once you upload it to the repo, replace the contents of this file with that document (or rename it).

---

## Quick reference (one-page summary)

For readers who don't need the full document, here's the methodological skeleton:

### Research questions

1. **RQ1**: Does an end-to-end audio-LLM (Qwen2-Audio-7B) outperform a cascaded ASR (Whisper-LV3) for Moroccan Darija, when both are fine-tuned with LoRA on the same data under the same protocol?
2. **RQ2**: How does data scale (10h → 30h) affect the comparison?
3. **RQ3**: How does each architecture handle code-switching (Arabic + Latin script alternation)?
4. **RQ4**: How does each architecture generalize to out-of-distribution (Casablanca-Morocco) Darija audio?

### Hypotheses (pre-registered before running experiments)

- **H1**: Qwen2-Audio-7B will outperform Whisper-LV3 at sufficient scale because of its much larger LLM backbone.
- **H2**: The gap will close with scale, with Qwen2-Audio benefiting more from additional data.
- **H3**: Both models will handle code-switching better than zero-shot Whisper.
- **H4**: OOD generalization will be similar for both models.

### Outcome vs hypotheses

| Hypothesis | Outcome      |
|-----------|---------------|
| H1: Qwen2-Audio outperforms Whisper | **FALSIFIED** — Whisper wins at every scale |
| H2: Gap closes with scale, Qwen benefits more | **PARTIALLY CONFIRMED** — gap closes (17.6pp → 8.4pp) but Qwen doesn't catch up |
| H3: Both handle CS better than ZS Whisper | **CONFIRMED** — CS gap drops from 27.9pp to 2-4pp |
| H4: OOD generalization similar | **NUANCED** — Whisper better absolute OOD WER, but worse relative OOD gap after FT |

### Controlled variables (identical for both models)

- Dataset: `atlasia/MoulSot-Full`, config `100-gt-2.5`
- Validation split: 2000 samples carved with `seed=42`
- Train set sampling: `seed=42` deterministic shuffle, accumulated to target hours (30h is a superset of 10h)
- Test set: publisher's official MoulSot test split (1962 utterances after audio-duration filter)
- OOD set: 500 samples from `UBC-NLP/Casablanca` (Morocco subset)
- LoRA: `r=32`, `α=32`, `dropout=0.1`, `target_modules="all-linear"`, rsLoRA
- Effective batch size: 16 (per-device 4 × grad-accum 4)
- BF16 precision, SDPA attention
- Marco-ASR adaptive LR + early stopping (patience=3, min_delta=0.5pp)
- Casablanca normalization (Talafha et al. 2024)
- Code-switching detection: reference contains both Arabic (U+0600-06FF) AND Latin characters
- Evaluation metric: corpus-level micro-average WER and CER (via `jiwer`)
- Statistical inference: 1000-resample paired bootstrap, 95% percentile CIs

### Variables that necessarily differ (architectural)

- **Marco-ASR algorithm**: Algorithm 1 (single adaptive LR) for the LLM-based Qwen2-Audio; Algorithm 2 (separate encoder + decoder LRs) for the encoder-decoder Whisper. Forcing Algorithm 1 on Whisper would yoke two components with very different optimization needs.
- **Audio input pipeline**: Qwen2-Audio uses a chat-template prompt + audio embedding; Whisper uses log-mel features + `forced_decoder_ids` for language/task conditioning.
- **Language conditioning**: Qwen2-Audio uses the prompt string; Whisper uses the `<|ar|>` language token (no Darija token exists in Whisper).
- **LoRA task type**: Qwen2-Audio uses `CAUSAL_LM`; Whisper uses no task type (the base `PeftModel` class, because `SEQ_2_SEQ_LM` is hardcoded for `input_ids` and Whisper takes `input_features`).

### Threats to validity (documented)

1. **Single-seed runs.** We trained each (model × scale) combination once. Variance across seeds is unmeasured. Mitigation: bootstrap CIs on the test-set evaluation give us a lower bound on result stability.
2. **MoulSot label noise.** The transcription quality of MoulSot-Full's `100-gt-2.5` config is not perfect; some references contain MSA-Darija blends that affect normalization-based WER computation. Both models are affected equally.
3. **OOD set characteristics.** Casablanca-Morocco's 500 samples may not represent the full OOD distribution; conclusions about OOD generalization are limited to this slice.
4. **Decoding strategy.** All evaluations use greedy decoding (num_beams=1). Beam search might shift relative orderings, especially on code-switched samples.
5. **Phase 1 doesn't measure downstream chatbot quality.** Our ASR-only conclusion may not extend to the conversational pipeline. Phase 2 will address this.

### Statistical protocol

- **Effect sizes**: WER point estimates with 95% percentile bootstrap CIs (1000 resamples)
- **Inter-model comparison**: 1000-resample paired bootstrap on the WER delta, on the same utterance indices
- **Significance threshold**: α = 0.05 (CI excludes 0)
- **Multiple testing**: No formal correction applied; we report all CIs and let readers judge

---

(To replace this stub with the full document, just paste the existing methodology markdown into this file.)
