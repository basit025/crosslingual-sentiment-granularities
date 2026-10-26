"""
Shared helpers for the camera-ready Stage 3 scripts.

Rebuilds the evaluation sets exactly as the original training code did and in the
same row order as results/confidence_data*.csv, and runs inference the same way
as the evaluation in the original training code (fp32, batch 64, padding to 256,
truncation='only_first'; mDeBERTa: slow tokenizer, no token_type_ids).

Row order of the existing prediction files (verified in Stage 1):
  finegrained : data/<split>.csv filtered by language
  binarized   : same, after dropping 3-star reviews
  triclass    : per language, seed-42 sample of n//2 reviews from stars 1, 2, 4, 5
                plus all 3-star reviews, then shuffled within the language
                (`.sample(frac=1, random_state=42)`)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

SEED = 42
LANGUAGES = ['en', 'de', 'es', 'fr', 'ja', 'zh']
MODELS = ['mbert', 'xlmr', 'mdeberta']
GRANS = ['finegrained', 'binarized', 'triclass']
MAX_LENGTH = 256
EVAL_BATCH_SIZE = 64

HF_NAMES = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'mdeberta': 'microsoft/mdeberta-v3-base',
}
PROB_COLS = {
    'finegrained': ['prob_1star', 'prob_2star', 'prob_3star', 'prob_4star', 'prob_5star'],
    'binarized': ['prob_neg', 'prob_pos'],
    'triclass': ['prob_neg', 'prob_neu', 'prob_pos'],
}
CONF_FILES = {
    'finegrained': 'results/confidence_data.csv',
    'binarized': 'results/confidence_data_binarized.csv',
    'triclass': 'results/confidence_data_triclass.csv',
}
TRICLASS_LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CR_DIR = Path('IMPROVEMENTS/camera_ready')
RESULTS_DIR = CR_DIR / 'results'


def checkpoint_path(model, gran, src, epoch):
    """Checkpoints are loaded in place, never copied."""
    if model == 'mdeberta':
        return Path('models') / f'mdeberta_{gran}' / src / f'epoch_{epoch}'
    return Path('models') / gran / model / src / f'epoch_{epoch}'


def _prepare(df):
    df = df.copy()
    df['review_title'] = df['review_title'].fillna('')
    df['product_category'] = df['product_category'].fillna('other')
    return df


def load_eval_sets(gran, split='test'):
    """Return {lang: DataFrame with columns review_body, review_title, product_category, label, lang_pos}
    in prediction-file order. `lang_pos` is the row's position among all reviews of that language in
    data/<split>.csv (used to take the balanced tri-class subset out of full-test-set predictions)."""
    raw = _prepare(pd.read_csv(f'data/{split}.csv'))
    sets = {}
    for lang in LANGUAGES:
        ld = raw[raw['language'] == lang]
        ld = ld.assign(lang_pos=np.arange(len(ld)))
        if gran == 'finegrained':
            d = ld.copy()
            d['label'] = d['stars'] - 1
        elif gran == 'binarized':
            d = ld[ld['stars'] != 3].copy()
            d['label'] = (d['stars'] >= 4).astype(int)
        else:
            star3 = ld[ld['stars'] == 3]
            n = len(star3)
            parts = [ld[ld['stars'] == s].sample(n=n // 2, random_state=SEED, replace=False) for s in (1, 2)]
            parts += [star3]
            parts += [ld[ld['stars'] == s].sample(n=n // 2, random_state=SEED, replace=False) for s in (4, 5)]
            d = pd.concat(parts).sample(frac=1, random_state=SEED).copy()
            d['label'] = d['stars'].map(TRICLASS_LABEL_MAP)
        sets[lang] = d.reset_index(drop=True)
    return sets


def load_tokenizer(model):
    if model == 'mdeberta':
        return AutoTokenizer.from_pretrained(HF_NAMES[model], use_fast=False)
    return AutoTokenizer.from_pretrained(HF_NAMES[model])


def encode(tokenizer, df, model):
    text_a = df['review_body'].astype(str).tolist()
    text_b = [f'{t} {c}' for t, c in zip(df['review_title'].astype(str), df['product_category'].astype(str))]
    enc = tokenizer(text_a, text_pair=text_b, truncation='only_first', max_length=MAX_LENGTH,
                    padding='max_length', return_tensors='pt')
    out = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
    if model != 'mdeberta' and 'token_type_ids' in enc:
        out['token_type_ids'] = enc['token_type_ids']
    return out


def load_model(model, gran, src, epoch):
    m = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path(model, gran, src, epoch)),
                                                           dtype=torch.float32)
    return m.to(DEVICE).eval()


@torch.inference_mode()
def predict_probs(model_obj, encoded):
    """Softmax probabilities for every example (fp32, batches of 64, original order)."""
    n = encoded['input_ids'].shape[0]
    probs = []
    for i in range(0, n, EVAL_BATCH_SIZE):
        batch = {k: v[i:i + EVAL_BATCH_SIZE].to(DEVICE) for k, v in encoded.items()}
        probs.append(torch.softmax(model_obj(**batch).logits.float(), dim=-1).cpu())
    return torch.cat(probs).numpy()


def prediction_frame(probs, labels, gran, model, src, tgt):
    """Same columns and column order as results/confidence_data*.csv."""
    pred = probs.argmax(axis=1)
    df = pd.DataFrame({
        'true_label': np.asarray(labels),
        'pred_label': pred,
        'correct': (pred == np.asarray(labels)).astype(int),
        'confidence': probs.max(axis=1),
    })
    for k, col in enumerate(PROB_COLS[gran]):
        df[col] = probs[:, k]
    df['model'] = model
    df['source_lang'] = src
    df['target_lang'] = tgt
    df['is_supervised'] = src == tgt
    return df


def selection_score(gran, true, pred):
    """Score used by the original training code to pick checkpoints: MAE x100 (lower) or macro-F1 (higher)."""
    from sklearn.metrics import f1_score
    true, pred = np.asarray(true), np.asarray(pred)
    if gran == 'binarized':
        return f1_score(true, pred, average='macro') * 100
    if gran == 'triclass':
        star = np.array([1.5, 3.0, 4.5])
        return np.abs(star[true] - star[pred]).mean() * 100
    return np.abs(true - pred).mean() * 100


def best_epoch(gran, scores):
    """First epoch with the best score (strict comparison, as in the original training code)."""
    best, best_ep = None, None
    for ep, s in enumerate(scores, start=1):
        if s is None:
            continue
        better = (best is None) or (s > best if gran == 'binarized' else s < best)
        if better:
            best, best_ep = s, ep
    return best_ep


def save_csv(df, path):
    """Never overwrite a results file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; use a new version name')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(obj, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; use a new version name')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
