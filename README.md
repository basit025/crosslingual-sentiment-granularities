# Cross-Lingual Sentiment Transfer Across Granularities

Code and results for the paper

> I. Hussain, A. Basit, and M. M. Rafiq, "Cross-Lingual Sentiment Transfer Across Granularities: Benchmarking and
> Explaining Multilingual Transformers," in *Proc. 23rd International Conference on Frontiers of Information
> Technology (FIT 2026)*, Islamabad, Pakistan, 2026 (to appear).

We fine-tune mBERT, XLM-R and mDeBERTa-v3 (base) on each of the six languages of the Multilingual Amazon Reviews Corpus
(MARC: EN, DE, ES, FR, JA, ZH) at three sentiment granularities (fine-grained 5-class, binarized 2-class, tri-class
3-class) and evaluate every model on all six target languages: 3 models × 6 sources × 3 granularities = 54 models,
324 (source, target, model, granularity) configurations.

## Contents

| Path | What it is |
|---|---|
| `IMPROVEMENTS/camera_ready/results/pairwise/` | Full 6×6 results (source rows, target columns) with 95% confidence intervals for every model, granularity and checkpoint-selection protocol |
| `IMPROVEMENTS/camera_ready/results/stage3/` | Epoch selected under each protocol, per-review predictions, bootstrap confidence intervals, paired significance tests |
| `IMPROVEMENTS/camera_ready/scripts/` | Scripts that produce these results and Fig. 1 (see the README there) |
| `results/` | Per-pair metrics, per-epoch validation scores and per-review predictions of the models in the paper |
| `IMPROVEMENTS/verify_table3_deep.py`, `IMPROVEMENTS/phase_2_*/` | Table III, Table V (language families), RQ2 normalized gaps, Table VIII (ECE) |

Protocols: *target_val* selects, for each source–target pair, the epoch with the best target-language validation score
(the paper's main protocol); *source_val* selects one epoch per model on the source-language validation set
(target-blind); *last_epoch* uses epoch 15; *published* denotes the predictions behind the paper's Table III
(mBERT and XLM-R: target-validation; mDeBERTa zero-shot: source-validation). Metrics: binarized accuracy;
fine-grained and tri-class MAE×100.

## Learning rates

Learning rate of every run, read from the saved training arguments of each checkpoint (all other settings as in the
paper):

| Model | Fine-grained | Binarized | Tri-class |
|---|---|---|---|
| mBERT | 2e-5 | 2e-5 | 2e-5 |
| XLM-R | 2e-5 | 2e-5 | 2e-5 |
| mDeBERTa-v3 | 3e-6 | 3e-6 | 2e-6 |

## Notes

1. The mDeBERTa zero-shot values of Tables III, V and VIII use source-validation checkpoints
   (`results/confidence_data*.csv.gz`); `results/*_results.csv` contain the target-validation results of all models.
   Table IV compares both protocols for all three models.
2. mDeBERTa binarized with FR or ZH as source is not reported (its intermediate checkpoints were not retained), so the
   FR and ZH rows of the binarized mDeBERTa pairwise tables are empty; the rows for these runs in
   `results/*binarized*.csv` are not used in the paper. The mDeBERTa binarized selection logs (EN/DE/ES/JA) were
   re-created by re-evaluating all 15 epochs on the validation sets (`IMPROVEMENTS/camera_ready/results/stage3/rebuilt_logs/`);
   they select the same epochs as `results/all_three_models_binarized.csv`.
3. The mDeBERTa tri-class rows of `results/confidence_data_triclass.csv.gz` (Table III) cover the full 5,000 test
   reviews per language; all other tri-class results use the balanced 3,000-review test sets.
4. `results/*_results.csv` were computed during training; the camera-ready analyses use per-review predictions
   re-computed from the same checkpoints. For mDeBERTa the per-pair MAE×100 differs by at most 0.22 (fine-grained
   zero-shot average under target-validation: 53.73 vs. 53.74 in Table IV). Fig. 1 shows the values of
   `results/*finegrained_results.csv`.
5. The Gap columns of Tables V and VI are differences of the rounded values printed in the tables (e.g. 55.7 − 46.1 =
   9.6); the scripts print the unrounded differences.

## Environment

Python 3.10; see `requirements.txt` (versions used for the camera-ready analyses). The large CSV files are
gzip-compressed; the scripts read them uncompressed, so first run `gunzip -k results/*.csv.gz IMPROVEMENTS/camera_ready/results/stage3/predictions/*.csv.gz`. The analyses ran on one NVIDIA RTX
A6000 (48 GB). Run all scripts from the repository root.

## License

Code: MIT (see `LICENSE`). MARC is subject to its own license and is not included.
