#!/usr/bin/env python3
"""
Stage 4.2: camera-ready copy of Fig. 1 (generate_paper_viz_part1.py, VIZ 2, fine-grained MAE x100).

Same data source and same cell values as the accepted figure:
load_data() and build_matrix() are copied unchanged from generate_paper_viz_part1.py, and the
cells are printed with the same format ('.1f'). Only the layout changes so the printed values
are readable at the width used in the paper: drawn at its final size (0.85 x IEEE text width),
one shared colour bar, no figure title (the LaTeX caption has it), larger annotation font.
The original script and figure are not modified.

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s4_2_fig1_heatmap.py
Outputs:
    IMPROVEMENTS/camera_ready/figures/three_way_mae_x100_finegrained_cr.png (600 dpi)
    IMPROVEMENTS/camera_ready/results/stage4/fig1_cell_values.csv (every printed value)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import RESULTS_DIR, save_csv  # noqa: E402

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']

# ─── CONSTANTS (as in generate_paper_viz_part1.py) ───
LANGUAGES = ['en', 'de', 'es', 'fr', 'ja', 'zh']
LANG_LABELS = ['EN', 'DE', 'ES', 'FR', 'JA', 'ZH']
MODELS = ['mbert', 'xlmr', 'mdeberta']
MODEL_LABELS = {'mbert': 'mBERT', 'xlmr': 'XLM-RoBERTa', 'mdeberta': 'mDeBERTa-v3'}

OUT_PNG = Path('IMPROVEMENTS/camera_ready/figures/three_way_mae_x100_finegrained_cr.png')
TEXT_WIDTH_IN = 7.16  # IEEEtran conference \textwidth
FIG_WIDTH_IN = 0.85 * TEXT_WIDTH_IN


def build_matrix(df, model_key, metric):
    sub = df[df['model'] == model_key]
    mat = np.full((6, 6), np.nan)
    for i, src in enumerate(LANGUAGES):
        for j, tgt in enumerate(LANGUAGES):
            row = sub[(sub['source_lang'] == src) & (sub['target_lang'] == tgt)]
            if len(row) > 0 and metric in row.columns:
                mat[i, j] = row[metric].values[0]
    return mat


def load_data():
    csvs = [
        'results/finegrained_results.csv',
        'results/binarized_results.csv',
        'results/triclass_results.csv',
        'results/mdeberta_finegrained_results.csv',
        'results/mdeberta_binarized_results.csv',
        'results/mdeberta_triclass_results.csv',
    ]
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


df_g = load_data()
df_g = df_g[df_g['granularity'] == 'finegrained']
mats = {m: build_matrix(df_g, m, 'mae_x100') for m in MODELS}
all_vals = np.concatenate([mat[~np.isnan(mat)] for mat in mats.values()])
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
assert not any(np.isnan(m).any() for m in mats.values())

if OUT_PNG.exists():
    raise FileExistsError(f'{OUT_PNG} exists; use a new version name')
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(FIG_WIDTH_IN, 2.15))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])
for idx, m in enumerate(MODELS):
    sns.heatmap(mats[m], annot=True, fmt='.1f', cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 6.5}, square=True, linewidths=0,
                xticklabels=LANG_LABELS, yticklabels=LANG_LABELS if idx == 0 else False,
                cbar=idx == 2, cbar_ax=cax if idx == 2 else None, ax=axes[idx])
    axes[idx].set_title(MODEL_LABELS[m], fontsize=8, fontweight='bold', pad=3)
    axes[idx].set_xlabel('Target language', fontsize=7, labelpad=1)
    axes[idx].tick_params(axis='both', labelsize=6.5, length=0, pad=1.5)
    axes[idx].tick_params(axis='y', rotation=0)
axes[0].set_ylabel('Source language', fontsize=7, labelpad=1)
cax.tick_params(labelsize=6.5, length=2, pad=1)
cax.set_ylabel('MAE$\\times$100', fontsize=7, labelpad=2)
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print(f'saved {OUT_PNG}')

rows = [{'model': m, 'source_lang': s, 'target_lang': t, 'is_supervised': s == t,
         'mae_x100_results_csv': mats[m][i, j], 'printed_value': f'{mats[m][i, j]:.1f}'}
        for m in MODELS for i, s in enumerate(LANGUAGES) for j, t in enumerate(LANGUAGES)]
save_csv(pd.DataFrame(rows), RESULTS_DIR / 'stage4' / 'fig1_cell_values.csv')
print(f'colour scale {vmin:.2f}-{vmax:.2f}; 108 printed values saved')
