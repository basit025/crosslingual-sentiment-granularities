"""
Stage 3.3: paired significance tests between models on the SAME test examples.

For each (granularity, protocol, source, target) and each comparison
(XLM-R vs mBERT, XLM-R vs mDeBERTa, mDeBERTa vs mBERT):
  binarized      : exact McNemar test on correct/incorrect (binomial test on the
                   discordant pairs); effect = accuracy(A) - accuracy(B)
  fine / tri     : paired bootstrap on per-example absolute-error differences
                   (10,000 resamples, seed 42); effect = MAE(A) - MAE(B) (x100,
                   0-based label distance). Chosen over Wilcoxon because it tests the
                   MAE difference itself (the reported metric), and the errors take
                   only a few integer values with many ties, which Wilcoxon handles poorly.
Holm-Bonferroni correction within each (granularity, protocol) family of
36 pairs x 3 comparisons = 108 tests (binarized: 84, as mDeBERTa has no FR/ZH-source runs).

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_3_paired_tests.py target_val source_val
Outputs (results/stage3/paired_tests/):
    paired_tests_<protocol>.csv   one row per test
    summary_<protocol>.csv        counts per granularity x comparison (zero-shot and supervised)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import GRANS, LANGUAGES, RESULTS_DIR, SEED, save_csv  # noqa: E402

N_BOOT = 10_000
ALPHA = 0.05
COMPARISONS = [('xlmr', 'mbert'), ('xlmr', 'mdeberta'), ('mdeberta', 'mbert')]
STAGE3 = RESULTS_DIR / 'stage3'
OUT = STAGE3 / 'paired_tests'


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_bootstrap(d, rng):
    """Two-sided p-value and 95% CI for mean(d), resampling examples (via counts of the distinct values)."""
    values, counts = np.unique(d, return_counts=True)
    n = len(d)
    obs = d.mean()
    means = rng.multinomial(n, counts / n, size=N_BOOT) @ values / n
    p = (1 + np.sum(np.abs(means - obs) >= abs(obs))) / (N_BOOT + 1) if obs != 0 else 1.0
    lo, hi = np.percentile(means, [2.5, 97.5])
    return p, lo, hi


def run(protocol):
    rows = []
    for gran in GRANS:
        rng = np.random.default_rng(SEED)
        df = pd.read_csv(STAGE3 / 'predictions' / f'confidence_data_{gran}_{protocol}.csv')
        family = []
        for src in LANGUAGES:
            for tgt in LANGUAGES:
                cell = df[(df.source_lang == src) & (df.target_lang == tgt)]
                by_model = {m: g.reset_index(drop=True) for m, g in cell.groupby('model')}
                for a, b in COMPARISONS:
                    if a not in by_model or b not in by_model:
                        continue  # mDeBERTa binarized has no FR/ZH-source runs
                    A, B = by_model[a], by_model[b]
                    assert (A.true_label.values == B.true_label.values).all()
                    row = {'protocol': protocol, 'granularity': gran, 'source_lang': src, 'target_lang': tgt,
                           'is_supervised': src == tgt, 'model_a': a, 'model_b': b, 'n': len(A)}
                    if gran == 'binarized':
                        ca, cb = A.correct.values.astype(bool), B.correct.values.astype(bool)
                        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
                        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                        row.update({'test': 'McNemar exact', 'metric': 'accuracy', 'metric_a': ca.mean() * 100,
                                    'metric_b': cb.mean() * 100, 'diff_a_minus_b': (ca.mean() - cb.mean()) * 100,
                                    'a_only_correct': n01, 'b_only_correct': n10, 'p_value': p,
                                    'a_better': ca.mean() > cb.mean()})
                    else:
                        ea = (A.true_label - A.pred_label).abs().values * 100.0
                        eb = (B.true_label - B.pred_label).abs().values * 100.0
                        p, lo, hi = paired_bootstrap(ea - eb, rng)
                        row.update({'test': 'paired bootstrap (abs error)', 'metric': 'mae_x100',
                                    'metric_a': ea.mean(), 'metric_b': eb.mean(), 'diff_a_minus_b': ea.mean() - eb.mean(),
                                    'diff_ci_low': lo, 'diff_ci_high': hi, 'p_value': p,
                                    'a_better': ea.mean() < eb.mean()})
                    family.append(row)
        adj = holm([r['p_value'] for r in family])
        for r, pa in zip(family, adj):
            r['p_holm'] = pa
            r['significant_holm'] = pa < ALPHA
        rows += family
        print(f'{protocol} {gran} done', flush=True)

    tests = pd.DataFrame(rows)
    save_csv(tests, OUT / f'paired_tests_{protocol}.csv')
    summary = []
    for (gran, a, b, sup), g in tests.groupby(['granularity', 'model_a', 'model_b', 'is_supervised']):
        summary.append({'protocol': protocol, 'granularity': gran, 'model_a': a, 'model_b': b,
                        'setting': 'supervised' if sup else 'zero_shot', 'n_pairs': len(g),
                        'a_lower_error_raw': int(g.a_better.sum()),
                        'a_significantly_better': int((g.a_better & g.significant_holm).sum()),
                        'b_significantly_better': int((~g.a_better & g.significant_holm).sum()),
                        'not_significant': int((~g.significant_holm).sum())})
    summary = pd.DataFrame(summary)
    save_csv(summary, OUT / f'summary_{protocol}.csv')
    print(summary.to_string())


for p in sys.argv[1:] or ['target_val', 'source_val']:
    run(p)
