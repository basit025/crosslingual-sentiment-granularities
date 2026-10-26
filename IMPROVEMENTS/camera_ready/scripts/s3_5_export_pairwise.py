"""
Stage 3.5: full pairwise results for the public repository.

For each protocol x granularity x model, writes a 6x6 matrix (source rows, target
columns) of the primary metric (binarized accuracy; fine-grained / tri-class MAE x100)
with bootstrap 95% CI bounds: for every target T the columns T, T_ci_low, T_ci_high.

Run from the project root (after s3_2):
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_5_export_pairwise.py published target_val source_val last_epoch
Output: IMPROVEMENTS/camera_ready/results/pairwise/<protocol>/<granularity>_<model>.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import GRANS, LANGUAGES, MODELS, RESULTS_DIR, save_csv  # noqa: E402

for protocol in sys.argv[1:]:
    ci = pd.read_csv(RESULTS_DIR / 'stage3' / 'bootstrap' / f'ci_cells_{protocol}.csv')
    for gran in GRANS:
        for model in MODELS:
            sub = ci[(ci.granularity == gran) & (ci.model == model)]
            mat = pd.DataFrame({'source_lang': LANGUAGES})
            for tgt in LANGUAGES:
                t = sub[sub.target_lang == tgt].set_index('source_lang').reindex(LANGUAGES)
                mat[tgt] = t['value'].round(2).values
                mat[f'{tgt}_ci_low'] = t['ci_low'].round(2).values
                mat[f'{tgt}_ci_high'] = t['ci_high'].round(2).values
            mat.insert(1, 'metric', sub.metric.iloc[0])
            save_csv(mat, RESULTS_DIR / 'pairwise' / protocol / f'{gran}_{model}.csv')
    print(f'{protocol}: exported')
