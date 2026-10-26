"""
Stage 3.1e: compact protocol table + checks of the paper's main conclusions.

Compact table: zero-shot (and supervised) averages per model and granularity under
  (a) target_val (original protocol, consistent for all three models)
  (b) source_val (target-blind)
  (c) last_epoch (target-blind reference, if computed)
with bootstrap 95% CI half-widths from s3_2. Metrics as in Table III: binarized accuracy,
fine-grained and tri-class MAE x100 (0-based label distance).
Binarized mDeBERTa covers the EN/DE/ES/JA sources (as in the paper's Table III).

Conclusion checks per protocol:
  C1 XLM-R best zero-shot (lowest MAE / highest accuracy) in every granularity
  C2 mDeBERTa competitive supervised, weaker than XLM-R zero-shot
  C3 neutral class hardest in tri-class (polar vs neutral F1, neutral-lowest pair count)
  C4 cross-family transfer harder than within Indo-European (all granularities)
  C5 RQ2 normalized supervised-to-zero-shot gaps

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_1e_compact_table_and_checks.py
Outputs (results/stage3/): compact_protocol_table.csv, conclusion_checks.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import GRANS, MODELS, RESULTS_DIR, save_csv  # noqa: E402

STAGE3 = RESULTS_DIR / 'stage3'
IE = {'en', 'de', 'es', 'fr'}
protocols = [p for p in ['target_val', 'source_val', 'last_epoch']
             if (STAGE3 / 'predictions' / f'confidence_data_finegrained_{p}.csv').exists()]


def metric_values(df, gran):
    return df['correct'] * 100.0 if gran == 'binarized' else (df['true_label'] - df['pred_label']).abs() * 100.0


def pair_table(df, gran):
    d = df.assign(v=metric_values(df, gran))
    return d.groupby(['model', 'source_lang', 'target_lang'])['v'].mean().reset_index()


# ── compact table ──
rows = []
for p in protocols:
    ci = pd.read_csv(STAGE3 / 'bootstrap' / f'ci_aggregates_{p}.csv')
    for r in ci.itertuples():
        rows.append({'protocol': p, 'model': r.model, 'granularity': r.granularity, 'setting': r.setting,
                     'sources': 'en/de/es/ja' if (r.model, r.granularity) == ('mdeberta', 'binarized') else 'all 6',
                     'n_pairs': r.n_pairs, 'metric': r.metric, 'value': r.value,
                     'ci_half_width': r.half_width, 'ci_low': r.ci_low, 'ci_high': r.ci_high})
compact = pd.DataFrame(rows)
save_csv(compact, STAGE3 / 'compact_protocol_table.csv')

# ── conclusion checks ──
checks = []


def add(p, cid, desc, value, holds):
    checks.append({'protocol': p, 'check': cid, 'description': desc, 'value': value, 'holds': holds})


for p in protocols:
    c = compact[(compact.protocol == p) & (compact.sources == 'all 6')]
    for gran in GRANS:
        zs = c[(c.granularity == gran) & (c.setting == 'zero_shot')].set_index('model').value
        sup = c[(c.granularity == gran) & (c.setting == 'supervised')].set_index('model').value
        best = zs.idxmax() if gran == 'binarized' else zs.idxmin()
        add(p, 'C1', f'{gran}: best zero-shot model', ', '.join(f'{m}={zs[m]:.2f}' for m in MODELS), best == 'xlmr')
        gap_sup = abs(sup['mdeberta'] - sup[sup.idxmax() if gran == 'binarized' else sup.idxmin()])
        add(p, 'C2a', f'{gran}: mDeBERTa supervised within 0.5 of the best supervised model',
            ', '.join(f'{m}={sup[m]:.2f}' for m in MODELS), gap_sup <= 0.5)
        weaker = zs['mdeberta'] < zs['xlmr'] if gran == 'binarized' else zs['mdeberta'] > zs['xlmr']
        add(p, 'C2b', f'{gran}: mDeBERTa weaker than XLM-R zero-shot (difference)',
            f"{zs['mdeberta'] - zs['xlmr']:+.2f}", bool(weaker))
    # C3 neutral class (tri-class, zero-shot)
    tri = pd.read_csv(STAGE3 / 'predictions' / f'confidence_data_triclass_{p}.csv')
    tri = tri[tri.source_lang != tri.target_lang]
    for m in MODELS:
        f1 = np.array([f1_score(g.true_label, g.pred_label, average=None, labels=[0, 1, 2]) * 100
                       for _, g in tri[tri.model == m].groupby(['source_lang', 'target_lang'])])
        polar, neu = ((f1[:, 0] + f1[:, 2]) / 2).mean(), f1[:, 1].mean()
        lowest = int((f1[:, 1] < np.minimum(f1[:, 0], f1[:, 2])).sum())
        add(p, 'C3', f'{m}: tri-class ZS polar F1, neutral F1, gap, neutral-lowest pairs',
            f'{polar:.2f}, {neu:.2f}, {polar - neu:.2f}, {lowest}/{len(f1)}', bool(polar > neu))
    # C4 cross-family (pooled over rows, as Table IV)
    for gran in GRANS:
        df = pd.read_csv(STAGE3 / 'predictions' / f'confidence_data_{gran}_{p}.csv')
        df = df[df.source_lang != df.target_lang]
        same = df.source_lang.isin(IE) & df.target_lang.isin(IE)
        for m in MODELS:
            v = metric_values(df[df.model == m], gran)
            s_, x_ = v[same[df.model == m]].mean(), v[~same[df.model == m]].mean()
            if gran == 'binarized':  # error rate, as in Table IV
                s_, x_ = 100 - s_, 100 - x_
            add(p, 'C4', f'{gran} {m}: within-IE vs cross-family ({"error %" if gran == "binarized" else "MAE x100"}), gap',
                f'{s_:.2f}, {x_:.2f}, {x_ - s_:+.2f}', bool(x_ > s_))
    # C5 normalized gaps (as phase_2_3/normalize_mae.py)
    for m in MODELS:
        vals = []
        for gran, k in [('binarized', 1), ('triclass', 2), ('finegrained', 4)]:
            zs = c[(c.granularity == gran) & (c.setting == 'zero_shot') & (c.model == m)].value.iloc[0]
            sp = c[(c.granularity == gran) & (c.setting == 'supervised') & (c.model == m)].value.iloc[0]
            vals.append((sp - zs) if gran == 'binarized' else (zs - sp) / k)
        add(p, 'C5', f'{m}: normalized ZS gap bin / tri / fine', ', '.join(f'{v:.2f}' for v in vals), None)

checks = pd.DataFrame(checks)
save_csv(checks, STAGE3 / 'conclusion_checks.csv')
pd.set_option('display.width', 220)
print(compact[compact.setting == 'zero_shot'].pivot_table(index=['granularity', 'model', 'sources'], columns='protocol',
                                                         values='value').round(2).to_string())
print(checks.to_string())
