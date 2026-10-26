"""
Stage 3.1a: epoch chosen for every (model, granularity, source, target) under each protocol.

  target_val : best epoch on the TARGET-language validation set (paper's protocol)
  source_val : best epoch on the SOURCE-language validation set (target-blind)
  last_epoch : epoch 15 (second target-blind reference)

Selection scores come from the existing logs (results/checkpoint_selection_logs/,
results/checkpoint_logs/); for mDeBERTa binarized, from the logs rebuilt by
s3_1b_rebuild_mdeberta_binarized_logs.py. Scores: MAE x100 for fine-grained and
tri-class, macro-F1 for binarized (same as the original selection). mDeBERTa binarized
covers the EN/DE/ES/JA sources, as in the paper.

Also records which epoch the existing per-example file (results/confidence_data*.csv)
was produced with, so later scripts reuse rows instead of re-running inference.

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_1a_select_epochs.py
Output: IMPROVEMENTS/camera_ready/results/stage3/selected_epochs.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import GRANS, LANGUAGES, MODELS, RESULTS_DIR, save_csv  # noqa: E402

REBUILT = RESULTS_DIR / 'stage3' / 'rebuilt_logs'
NOT_REPORTED = {('mdeberta', 'binarized', 'fr'), ('mdeberta', 'binarized', 'zh')}


def log_path(model, gran, src):
    if model == 'mdeberta' and gran == 'binarized':
        return REBUILT / f'checkpoint_selection_mdeberta_{src}_binarized.json'
    if model == 'mdeberta':
        return Path('results/checkpoint_logs') / f'checkpoint_selection_mdeberta_{src}_{gran}.json'
    suffix = '' if gran == 'finegrained' else f'_{gran}'
    return Path('results/checkpoint_selection_logs') / f'checkpoint_selection_{model}_{src}{suffix}.json'


rows = []
for gran in GRANS:
    res = pd.read_csv(f'results/all_three_models_{gran}.csv')
    for model in MODELS:
        for src in LANGUAGES:
            if (model, gran, src) in NOT_REPORTED:
                continue
            log = json.loads(log_path(model, gran, src).read_text())
            src_epoch = log[src]['best_epoch']
            csv_diag = int(res[(res.model == model) & (res.source_lang == src) & (res.target_lang == src)].best_epoch.iloc[0])
            for tgt in LANGUAGES:
                csv_epoch = int(res[(res.model == model) & (res.source_lang == src) & (res.target_lang == tgt)].best_epoch.iloc[0])
                # target-val epoch: the published one from the results CSV
                tv = csv_epoch
                # epoch behind the existing per-example rows (mDeBERTa: source-validation epoch)
                conf_epoch = csv_diag if model == 'mdeberta' else csv_epoch
                rows.append({
                    'model': model, 'granularity': gran, 'source_lang': src, 'target_lang': tgt,
                    'is_supervised': src == tgt,
                    'epoch_target_val': tv, 'epoch_source_val': src_epoch, 'epoch_last': 15,
                    'epoch_existing_confidence_data': conf_epoch,
                    'results_csv_best_epoch': csv_epoch,
                    'log_target_val_epoch': log[tgt]['best_epoch'],
                })

out = pd.DataFrame(rows)
# consistency checks against the published results
mism_tv = out[out.log_target_val_epoch != out.results_csv_best_epoch]
print(f'log target-val epoch != results CSV best_epoch: {len(mism_tv)}')
if len(mism_tv):
    print(mism_tv[['model', 'granularity', 'source_lang', 'target_lang', 'log_target_val_epoch', 'results_csv_best_epoch']].to_string())
sup = out[out.is_supervised]
print('supervised: source-val epoch == target-val epoch for all runs:',
      bool((sup.epoch_source_val == sup.epoch_target_val).all()))
zs = out[~out.is_supervised]
print('ZS pairs where source-val epoch differs from target-val epoch:')
print(zs.assign(diff=zs.epoch_source_val != zs.epoch_target_val).groupby(['granularity', 'model'])['diff'].sum().unstack().to_string())

VERSION = sys.argv[1] if len(sys.argv) > 1 else ''
save_csv(out, RESULTS_DIR / 'stage3' / f'selected_epochs{VERSION}.csv')
