# Camera-ready analysis scripts

These scripts produce the checkpoint-selection protocols, confidence intervals, paired tests, pairwise tables and
Fig. 1 of the paper. Run them **from the repository root** with the packages of `requirements.txt`. They need the MARC
data in `data/` and the 54 fine-tuned models (15 epoch checkpoints each) in `models/`, neither of which is distributed,
so the outputs are included in `IMPROVEMENTS/camera_ready/results/`. Every output file is written only if it does not
exist yet. Seed 42 everywhere; bootstrap with 10,000 resamples.

| Order | Script | What it does | Outputs (under `IMPROVEMENTS/camera_ready/results/`) |
|---|---|---|---|
| — | `cr_common.py` | Shared helpers: test/validation sets in the row order of the prediction files, sentence-pair tokenisation (256 tokens), batched fp32 inference, selection score | — |
| 1 | `s3_1b_rebuild_mdeberta_binarized_logs.py` | Re-evaluates all 15 epochs of the mDeBERTa binarized models (EN/DE/ES/JA sources) on the six validation sets (their original selection logs were overwritten) | `stage3/rebuilt_logs/` |
| 2 | `s3_1a_select_epochs.py` | Epoch of every (model, granularity, source, target) under target-validation, source-validation and last epoch | `stage3/selected_epochs.csv` |
| 3 | `s3_1c_inference.py target_val source_val`, then `s3_1c_inference.py last_epoch` | Per-review test predictions for every protocol (existing predictions are reused when the checkpoint is the same) | `stage3/predictions/`, `stage3/provenance_*.csv` |
| 4 | `s3_2_bootstrap_ci.py published target_val source_val last_epoch` | 95% bootstrap CIs per pair and for the zero-shot / supervised averages | `stage3/bootstrap/` |
| 5 | `s3_3_paired_tests.py target_val source_val last_epoch` | Paired tests on identical reviews (McNemar exact for binarized; paired bootstrap of absolute errors for fine-grained / tri-class), Holm–Bonferroni | `stage3/paired_tests/` |
| 6 | `s3_1e_compact_table_and_checks.py` | Checks of the main findings; this repository keeps C3 (neutral vs. polar F1: Table VI, Sec. V-D) and C4 (within- vs. cross-family: Sec. V-C) | `stage3/conclusion_checks.csv` |
| 7 | `s3_5_export_pairwise.py published target_val source_val last_epoch` | 6×6 tables (source rows, target columns) with CI bounds | `pairwise/<protocol>/<granularity>_<model>.csv` |
| 8 | `s4_2_fig1_heatmap.py` | Fig. 1 from `results/*finegrained_results.csv` | `../figures/three_way_mae_x100_finegrained_cr.png`, `stage4/fig1_cell_values.csv` |

mDeBERTa binarized covers the EN/DE/ES/JA sources, as in the paper (pairwise tables: empty FR/ZH rows).
`conclusion_checks.csv` in this repository keeps checks C3 and C4.

Definitions: target-validation = best epoch on the target-language validation set for each pair; source-validation =
one epoch per model, best on the source-language validation set; last epoch = epoch 15. Selection score: validation
MAE (fine-grained, tri-class) or macro-F1 (binarized); ties go to the earliest epoch. Metrics: binarized accuracy;
fine-grained and tri-class MAE×100 on 0-based labels; tri-class evaluated on the balanced 3,000-review test set.
