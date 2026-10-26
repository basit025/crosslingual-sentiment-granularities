"""
Stage 3.1c: per-example test predictions under each checkpoint-selection protocol.

For every (model, granularity, source, target) and protocol, the epoch comes from
selected_epochs.csv (s3_1a). Existing rows of results/confidence_data*.csv are reused
when they were produced with exactly that checkpoint; otherwise the checkpoint is
loaded in place and run on the target test set (same pipeline as the original training code).

New inference is saved per checkpoint the moment it is computed
(results/stage3/new_inference/<gran>/<model>_<src>_epoch<N>.csv, all needed targets),
so the job can be resumed. Then one file per granularity and protocol is assembled
in the confidence_data*.csv format (same columns):
    results/stage3/predictions/confidence_data_<gran>_<protocol>.csv
plus a provenance table saying, for every cell, which epoch and which source was used.

Tri-class: all models are evaluated on the balanced 3,000-review test set per language
(mDeBERTa's existing rows cover the full 5,000 reviews; the balanced subset is taken out).

Run from the project root:
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_1c_inference.py target_val source_val
    .venv/bin/python IMPROVEMENTS/camera_ready/scripts/s3_1c_inference.py last_epoch
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cr_common import (CONF_FILES, GRANS, LANGUAGES, MODELS, RESULTS_DIR, encode, load_eval_sets,  # noqa: E402
                       load_model, load_tokenizer, predict_probs, prediction_frame, save_csv)

PROTOCOL_COLS = {'target_val': 'epoch_target_val', 'source_val': 'epoch_source_val', 'last_epoch': 'epoch_last'}
protocols = sys.argv[1:] or ['target_val', 'source_val']
assert all(p in PROTOCOL_COLS for p in protocols), protocols

STAGE3 = RESULTS_DIR / 'stage3'
NEW_DIR = STAGE3 / 'new_inference'
PRED_DIR = STAGE3 / 'predictions'
sel = pd.read_csv(STAGE3 / 'selected_epochs.csv')


def chunk_path(model, gran, src, epoch):
    return NEW_DIR / gran / f'{model}_{src}_epoch{epoch}.csv'


# 1) plan: which cells need new inference
plan = []
for p in protocols:
    for r in sel.itertuples():
        epoch = int(getattr(r, PROTOCOL_COLS[p]))
        existing = r.epoch_existing_confidence_data
        reuse = pd.notna(existing) and int(existing) == epoch
        plan.append({'protocol': p, 'model': r.model, 'granularity': r.granularity, 'source_lang': r.source_lang,
                     'target_lang': r.target_lang, 'epoch': epoch,
                     'source': 'reused results/confidence_data' if reuse else 'new inference'})
plan = pd.DataFrame(plan)
todo = (plan[plan.source == 'new inference']
        .groupby(['granularity', 'model', 'source_lang', 'epoch'])['target_lang']
        .apply(lambda s: sorted(set(s))).reset_index())
print(f'cells: {len(plan)} | new-inference cells: {(plan.source == "new inference").sum()} | '
      f'checkpoints to load: {len(todo)}', flush=True)

# 2) run new inference, one checkpoint at a time, saving immediately (resumable)
t0 = time.time()
test_sets = {g: load_eval_sets(g) for g in GRANS}
encoded_cache = {}
for i, r in enumerate(todo.itertuples(), start=1):
    out_path = chunk_path(r.model, r.granularity, r.source_lang, r.epoch)
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=['target_lang']).target_lang.unique())
        if set(r.target_lang) <= done:
            continue
        raise FileExistsError(f'{out_path} exists but lacks targets {set(r.target_lang) - done}; use a new version')
    model_obj = load_model(r.model, r.granularity, r.source_lang, r.epoch)
    frames = []
    for tgt in r.target_lang:
        key = (r.model == 'mdeberta', r.model, r.granularity, tgt)
        if key not in encoded_cache:
            encoded_cache[key] = encode(load_tokenizer(r.model), test_sets[r.granularity][tgt], r.model)
        probs = predict_probs(model_obj, encoded_cache[key])
        frames.append(prediction_frame(probs, test_sets[r.granularity][tgt]['label'].values,
                                       r.granularity, r.model, r.source_lang, tgt))
    save_csv(pd.concat(frames, ignore_index=True), out_path)
    del model_obj
    print(f'[{i}/{len(todo)}] {r.granularity} {r.model} {r.source_lang} epoch {r.epoch} -> '
          f'{len(r.target_lang)} targets ({(time.time() - t0) / 60:.1f} min)', flush=True)

# 3) assemble one confidence_data-format file per granularity and protocol
for gran in GRANS:
    conf = pd.read_csv(CONF_FILES[gran])
    for p in protocols:
        frames = []
        for model in MODELS:
            for src in LANGUAGES:
                for tgt in LANGUAGES:
                    cell = plan[(plan.protocol == p) & (plan.granularity == gran) & (plan.model == model) &
                                (plan.source_lang == src) & (plan.target_lang == tgt)].iloc[0]
                    if cell.source == 'new inference':
                        c = pd.read_csv(chunk_path(model, gran, src, cell.epoch))
                        c = c[c.target_lang == tgt]
                    else:
                        c = conf[(conf.model == model) & (conf.source_lang == src) & (conf.target_lang == tgt)]
                        if model == 'mdeberta' and gran == 'triclass':
                            # existing mDeBERTa tri-class rows cover the full test set (data/test.csv order);
                            # keep the balanced subset, in the same order as the other models
                            c = c.reset_index(drop=True).iloc[test_sets[gran][tgt]['lang_pos'].values]
                    c = c.reset_index(drop=True)
                    assert (c.true_label.values == test_sets[gran][tgt]['label'].values).all(), (gran, p, model, src, tgt)
                    frames.append(c[conf.columns])
        save_csv(pd.concat(frames, ignore_index=True), PRED_DIR / f'confidence_data_{gran}_{p}.csv')
        print(f'assembled {gran} {p}', flush=True)

save_csv(plan, STAGE3 / f'provenance_{"_".join(protocols)}.csv')
print(f'total {(time.time() - t0) / 60:.1f} min')
