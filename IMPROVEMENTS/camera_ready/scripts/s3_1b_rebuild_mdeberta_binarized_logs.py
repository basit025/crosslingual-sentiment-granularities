"""
Stage 3.1b: rebuild the mDeBERTa binarized checkpoint-selection logs.

The results/checkpoint_logs/checkpoint_selection_mdeberta_<lang>_binarized.json files were
overwritten with empty files. This script re-evaluates every epoch (1-15) of the mDeBERTa
binarized models with EN, DE, ES and JA as source (the sources reported in the paper) on all
six binarized validation sets, exactly like the original checkpoint selection (macro-F1, strict '>' so the
earliest best epoch wins).

New logs are written to IMPROVEMENTS/camera_ready/results/stage3/rebuilt_logs/,
never into results/. Per-epoch scores (macro-F1, accuracy) also go to a CSV.

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_1b_rebuild_mdeberta_binarized_logs.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import (LANGUAGES, RESULTS_DIR, best_epoch, encode, load_eval_sets, load_model,  # noqa: E402
                       load_tokenizer, predict_probs, save_csv, save_json, selection_score)

GRAN = 'binarized'
SOURCES = ['en', 'de', 'es', 'ja']
EPOCHS = 15
OUT = RESULTS_DIR / 'stage3' / 'rebuilt_logs'

val_sets = load_eval_sets(GRAN, split='validation')
tok = load_tokenizer('mdeberta')
encoded = {lang: encode(tok, val_sets[lang], 'mdeberta') for lang in LANGUAGES}

score_rows = []
t0 = time.time()
for src in SOURCES:
    for epoch in range(1, EPOCHS + 1):
        model = load_model('mdeberta', GRAN, src, epoch)
        for tgt in LANGUAGES:
            pred = predict_probs(model, encoded[tgt]).argmax(axis=1)
            true = val_sets[tgt]['label'].values
            score_rows.append({'model': 'mdeberta', 'granularity': GRAN, 'source_lang': src, 'epoch': epoch,
                               'target_lang': tgt, 'val_macro_f1': selection_score(GRAN, true, pred),
                               'val_accuracy': (pred == true).mean() * 100})
        del model
        print(f'{src} epoch {epoch} done ({(time.time() - t0) / 60:.1f} min)', flush=True)

    scores = pd.DataFrame([r for r in score_rows if r['source_lang'] == src])
    log = {}
    for tgt in LANGUAGES:
        f1s = scores[scores.target_lang == tgt].sort_values('epoch')['val_macro_f1'].tolist()
        ep = best_epoch(GRAN, f1s)
        log[tgt] = {'best_epoch': ep, 'best_val_f1': f1s[ep - 1], 'epoch_f1s': f1s}
    save_json(log, OUT / f'checkpoint_selection_mdeberta_{src}_binarized.json')

save_csv(pd.DataFrame(score_rows), OUT / 'mdeberta_binarized_val_scores_all_epochs.csv')
print(f'total {(time.time() - t0) / 60:.1f} min')
