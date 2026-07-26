# EasyOCR Fine-Tune on CGHD Handwritten Schematic Labels — Final Report

## Objective

EasyOCR's stock `english_g2` recognizer misreads handwritten component
designators on CGHD hand-drawn schematics (e.g. `R1` → `Ral`). This project
fine-tuned `english_g2` on real CGHD (crop, transcription) ground truth to
close that gap, following the pinned-commit trainer workflow in
`JaidedAI/EasyOCR@363afb184047ce452e436f4224f3098422df872e`.

## Dataset

- Source: `data/cghd-zenodo-12.zip`, built via
  `src/imgschem/datasets/ocr_finetune_data.py::build_ocr_finetune_dataset`.
- 39,246 train / 19,730 val (image crop, transcription) pairs, split by
  drafter (no handwriting style shared across train/val).
- 12,712 untranscribed boxes and 2,616 rotated boxes excluded by design
  (documented gaps, not blockers).
- Character set: 98 characters — digits, English letters, common symbols,
  Ω (GREEK CAPITAL LETTER OMEGA, U+03A9), µ (MICRO SIGN, U+00B5), and a
  handful of German umlauts from unrelated title-block text.

## Environment issues found and fixed along the way

These are noted because they materially affected what "verified" means for
this run — none were guessed past, all were confirmed against real source
or real error output before being patched:

1. **Architecture mismatch (config said ResNet, `english_g2.pth` is
   actually VGG).** Confirmed by tracing the real `easyocr.easyocr`
   routing logic (`english_g2` → `recog_network='generation2'` →
   `easyocr.model.vgg_model`) and independently by the pretrained
   checkpoint's own key structure (a flat numbered `nn.Sequential`, not
   ResNet's named submodules). Config corrected to
   `FeatureExtraction: VGG`.
2. **Dataset folder layout.** EasyOCR's `hierarchical_dataset`/`OCRDataset`
   require images and `labels.csv` in the *same* leaf directory, no
   `images/` subfolder — confirmed from the real trainer source, not the
   sample config's naming. `ocr_finetune_data.py` and its test were
   corrected to write flat.
3. **`.next()` on a DataLoader iterator** — removed in current PyTorch;
   patched to `next(iterator)` in the pinned trainer's `dataset.py`.
4. **Device mismatch in `test.py`'s `validation()`** — `text_for_loss`,
   `length_for_loss`, `preds_size` were never moved to `cuda`, unlike the
   equivalent line in `train.py`'s own training loop. Patched to match.
5. **Post-validation `CUDNN_STATUS_INTERNAL_ERROR`** — no
   `torch.cuda.empty_cache()` anywhere in the pinned trainer; on a 4GB
   card, memory fragmentation after a full validation pass reliably broke
   the next LSTM forward pass. Fixed by adding `empty_cache()` after each
   validation call.
6. **Windows DataLoader worker respawn exhausting the page file** — every
   validation pass spawned fresh worker subprocesses that each reloaded
   PyTorch's CUDA DLLs from scratch; fixed by setting `workers: 0`.
7. **`prefetch_factor=512` hardcoded on the validation DataLoader**, invalid
   once `workers: 0`; made conditional on `num_workers > 0`.
8. **Character-set typo: Greek mu (μ, U+03BC) vs. the real data's micro
   sign (µ, U+00B5).** These are visually identical, different Unicode
   codepoints. The config had the wrong one; 2,137 real training rows and
   406 real val rows containing µ were being silently filtered out of
   training as a result. Fixed by correcting `symbol` in
   `ocr_finetune_config.yaml`. (Ω was checked for the equivalent
   OHM SIGN/GREEK OMEGA duplicate-codepoint trap and found clean — all
   1,860 real occurrences already use the correct codepoint.)

## Training

- Base checkpoint: `english_g2.pth`, fine-tuned via `FT=True,
  new_prediction=True` (pretrained FeatureExtraction/SequenceModeling
  weights preserved byte-for-byte, confirmed by an explicit checkpoint-load
  sanity check before training; only the Prediction layer reinitialized to
  the new 98-char vocabulary).
- Hardware: NVIDIA RTX 3050 Laptop GPU, 4GB VRAM.
- `batch_size: 32`, `num_iter: 20000`, `valInterval: 1000`.
- Two full training runs were completed:
  - **Run 1** (µ character-set bug still present): best designator
    accuracy, checkpoint at iteration 9000.
  - **Run 2** (µ bug fixed, 2,543 additional real training/val rows
    included): checkpoint at iteration 13000 selected by the trainer's own
    `best_accuracy` tracking.
- Both runs showed the same overfitting shape: training loss →
  near-zero by iteration 20000 while validation loss climbed steadily past
  its minimum around iteration 3000–6000. **The final-iteration checkpoint
  was not used in either run** — the mid-run "best accuracy" checkpoint was
  used instead, per the task's own guidance to revise based on the real
  val curve rather than run to completion blindly.

## Results — held-out drafters, 19,730 val samples

### Overall

| | Pretrained `english_g2` | Fine-tuned (Run 1) | Fine-tuned (Run 2, shipped) |
|---|---|---|---|
| CER | 0.5522 | 0.1783 | 0.1771 |
| WER | 0.8487 | 0.3583 | 0.3598 |

### Designator spot-check (R1 / U1 / L1 / I2 — the originally-reported failures)

| Designator | Pretrained | Run 1 | Run 2 (shipped) |
|---|---|---|---|
| R1 | 46/181 (25%) | 174/181 (96%) | 155/181 (86%) |
| U1 | 21/78 (27%) | 73/78 (94%) | 70/78 (90%) |
| L1 | 12/31 (39%) | 25/31 (81%) | 16/31 (52%) |
| I2 | 0/8 (0%) | 7/8 (88%) | 7/8 (88%) |

### Special-character labels (Ω / µ, common in component values like "10Ω", "4.7µF")

| | Pretrained | Run 1 (µ bug) | Run 2 (µ fixed) |
|---|---|---|---|
| Ω CER (873 samples) | not evaluable* | 0.3919 | 0.3705 |
| µ exact-match (406 samples) | 0% (unsupported)* | 0% (architecturally impossible) | **78.6%** |
| µ CER | — | — | 0.0626 |

\* Pretrained `english_g2` uses its own 96-character vocabulary and
cannot produce Ω/µ at all; its errors on these samples are a vocabulary
gap, not purely a handwriting-recognition failure.

## Decision: which checkpoint shipped, and why

**Run 2's iteration-13000 checkpoint was packaged and shipped**
(`schematic_designator_ocr_v2`), trading some designator accuracy (notably
L1, 81%→52%) for making µ-containing value labels readable at all
(0%→78.6%). This is a real trade-off, not a strict improvement — Run 1's
checkpoint is preserved at
`saved_models\cghd_ocr_finetune_v1_run1_backup\` and can be packaged
identically if L1/R1 accuracy turns out to matter more in practice than
µ support.

The L1 regression was checked across three checkpoints within Run 2
(iterations 6000, 10000, 13000) and was consistent throughout — this rules
out "wrong checkpoint pick" and points instead to the changed training-data
composition (2,543 additional real rows) shifting the whole run's dynamics,
not a fixable checkpoint-selection issue.

## Known limitations (for future work, not blockers)

- **L1 designator accuracy (52%) is worse than it was before the µ fix
  (81%).** Candidate fixes: a longer run, a different train/val shuffle, or
  explicit oversampling of short designator-style labels alongside the
  µ-value-labels so neither is traded off against the other.
- **Ω-containing value labels remain weak (CER ~0.37–0.40)** in both runs,
  independent of the µ bug — most likely genuine data scarcity (1,860 real
  Ω occurrences vs. tens of thousands of digit/letter occurrences).
  Oversampling Ω crops is the natural next experiment.
- **Free-text labels longer than `batch_max_length: 34` characters are
  excluded from training entirely** (by design, per the original config's
  own justification) and will not be read correctly regardless of
  checkpoint.
- Rotated text boxes (4.1% of the corpus) and untranscribed boxes remain
  out of scope, as originally documented.

## Deliverable

Packaged per `EasyOCR`'s `custom_model.md` convention, verified by an
actual `easyocr.Reader(recog_network=...)` load (not assumed):

- `schematic_designator_ocr_v2.pth`
- `schematic_designator_ocr_v2.yaml`
- `schematic_designator_ocr_v2.py`
