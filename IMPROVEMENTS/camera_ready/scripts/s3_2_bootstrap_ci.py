"""
Stage 3.2: bootstrap 95% confidence intervals.

Non-parametric bootstrap over TEST EXAMPLES, percentile 95% CI, seed 42.
Within each replicate, ONE resample of each target-language test set is drawn
and applied to every (model, source) cell with that target, so the aggregate
averages (mean over pairs, as in Table III) are bootstrapped jointly.

Per-example value: correct (x100) for binarized accuracy, |true - pred| (x100)
for fine-grained and tri-class MAE (0-based label distance, as in Table III).

Prediction sets ("protocols"):
  published  : results/confidence_data*.csv as used for the accepted Table III
               (mBERT/XLM-R target-val; mDeBERTa source-val, binarized en/de/es/ja
               only, tri-class on the full test set)
  target_val, source_val, last_epoch : results/stage3/predictions/*.csv (s3_1c)

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_2_bootstrap_ci.py published target_val source_val
Outputs (results/stage3/bootstrap/):
    ci_cells_<protocol>.csv      CI for every (model, gran, source, target)
    ci_aggregates_<protocol>.csv CI for zero-shot and supervised averages per model and gran
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import CONF_FILES, GRANS, LANGUAGES, MODELS, RESULTS_DIR, SEED, save_csv  # noqa: E402

N_BOOT = 10_000
CHUNK = 1_000
STAGE3 = RESULTS_DIR / 'stage3'
OUT = STAGE3 / 'bootstrap'


def load_predictions(gran, protocol):
    if protocol == 'published':
        return pd.read_csv(CONF_FILES[gran])
    return pd.read_csv(STAGE3 / 'predictions' / f'confidence_data_{gran}_{protocol}.csv')


def per_example_value(df, gran):
    if gran == 'binarized':
        return df['correct'].values * 100.0
    return (df['true_label'] - df['pred_label']).abs().values * 100.0


def run(protocol):
    cell_rows, agg_rows = [], []
    for gran in GRANS:
        rng = np.random.default_rng(SEED)
        df = load_predictions(gran, protocol)
        metric = 'accuracy' if gran == 'binarized' else 'mae_x100'
        # replicate values per cell: {(model, src, tgt): array(N_BOOT)}
        reps, point, n_test = {}, {}, {}
        for tgt in LANGUAGES:
            sub = df[df.target_lang == tgt]
            cells = [(m, s) for m in MODELS for s in LANGUAGES
                     if len(sub[(sub.model == m) & (sub.source_lang == s)])]
            # cells with the same target share the test set only if they have the same size
            sizes = {c: len(sub[(sub.model == c[0]) & (sub.source_lang == c[1])]) for c in cells}
            for n in sorted(set(sizes.values())):
                group = [c for c in cells if sizes[c] == n]
                V = np.stack([per_example_value(sub[(sub.model == m) & (sub.source_lang == s)], gran)
                              for m, s in group], axis=1)  # n x cells
                out = np.empty((N_BOOT, len(group)))
                for start in range(0, N_BOOT, CHUNK):
                    w = rng.multinomial(n, np.full(n, 1.0 / n), size=min(CHUNK, N_BOOT - start)).astype(np.float64)
                    out[start:start + len(w)] = w @ V / n
                for j, (m, s) in enumerate(group):
                    reps[(m, s, tgt)] = out[:, j]
                    point[(m, s, tgt)] = V[:, j].mean()
                    n_test[(m, s, tgt)] = n
        for (m, s, t), r in reps.items():
            lo, hi = np.percentile(r, [2.5, 97.5])
            cell_rows.append({'protocol': protocol, 'model': m, 'granularity': gran, 'source_lang': s, 'target_lang': t,
                              'is_supervised': s == t, 'metric': metric, 'value': point[(m, s, t)],
                              'ci_low': lo, 'ci_high': hi, 'half_width': (hi - lo) / 2,
                              'n_test': n_test[(m, s, t)]})
        for m in MODELS:
            for setting, keep in [('zero_shot', lambda s, t: s != t), ('supervised', lambda s, t: s == t)]:
                keys = [k for k in reps if k[0] == m and keep(k[1], k[2])]
                if not keys:
                    continue
                agg = np.mean([reps[k] for k in keys], axis=0)
                val = np.mean([point[k] for k in keys])
                lo, hi = np.percentile(agg, [2.5, 97.5])
                agg_rows.append({'protocol': protocol, 'model': m, 'granularity': gran, 'setting': setting,
                                 'metric': metric, 'n_pairs': len(keys), 'value': val, 'ci_low': lo, 'ci_high': hi,
                                 'half_width': (hi - lo) / 2})
        print(f'{protocol} {gran} done', flush=True)
    save_csv(pd.DataFrame(cell_rows), OUT / f'ci_cells_{protocol}.csv')
    save_csv(pd.DataFrame(agg_rows), OUT / f'ci_aggregates_{protocol}.csv')
    print(pd.DataFrame(agg_rows).round(2).to_string())


for p in sys.argv[1:] or ['published']:
    run(p)
