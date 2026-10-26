import pandas as pd
import os

def compute_mae(df):
    return (df['true_label'] - df['pred_label']).abs().mean() * 100

def compute_error_rate(df):
    return (1.0 - df['correct'].mean()) * 100

def get_family_type(src, tgt):
    ie_langs = {'en', 'de', 'es', 'fr'}
    if src == tgt:
        return 'supervised'
    if src in ie_langs and tgt in ie_langs:
        return 'same'
    return 'cross'

def is_ja_zh_source(src, tgt):
    return src in {'ja', 'zh'} and tgt in {'en', 'de', 'es', 'fr'}

def is_ja_zh_target(src, tgt):
    return src in {'en', 'de', 'es', 'fr'} and tgt in {'ja', 'zh'}

def main():
    os.makedirs('IMPROVEMENTS/phase_2_1', exist_ok=True)
    
    # 1. Binarized Error Rates
    df_bin = pd.read_csv('results/confidence_data_binarized.csv')
    df_bin['family_type'] = df_bin.apply(lambda row: get_family_type(row['source_lang'], row['target_lang']), axis=1)
    df_bin_zs = df_bin[df_bin['family_type'] != 'supervised']
    
    bin_results = {}
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df_bin_zs[df_bin_zs['model'] == model]
        same_df = m_df[m_df['family_type'] == 'same']
        cross_df = m_df[m_df['family_type'] == 'cross']
        
        same_err = compute_error_rate(same_df) if len(same_df) > 0 else float('nan')
        cross_err = compute_error_rate(cross_df) if len(cross_df) > 0 else float('nan')
        
        bin_results[model] = {'Same': same_err, 'Cross': cross_err, 'Gap': cross_err - same_err}

    # 2. Fine-Grained MAE
    df_fg = pd.read_csv('results/confidence_data.csv')
    df_fg['family_type'] = df_fg.apply(lambda row: get_family_type(row['source_lang'], row['target_lang']), axis=1)
    df_fg_zs = df_fg[df_fg['family_type'] != 'supervised']
    
    fg_results = {}
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df_fg_zs[df_fg_zs['model'] == model]
        same_df = m_df[m_df['family_type'] == 'same']
        cross_df = m_df[m_df['family_type'] == 'cross']
        
        same_mae = compute_mae(same_df)
        cross_mae = compute_mae(cross_df)
        
        fg_results[model] = {'Same': same_mae, 'Cross': cross_mae, 'Gap': cross_mae - same_mae}

    # Output for Table IV
    print("="*50)
    print("RECOMPUTED TABLE IV (All 30 Zero-Shot Pairs)")
    print("="*50)
    print(f"{'Model':<10} | {'Binarized (Error%)':<25} | {'Fine-Grained (MAE)':<25}")
    print(f"{'':<10} | {'Same':<7} {'Cross':<7} {'Gap':<7} | {'Same':<7} {'Cross':<7} {'Gap':<7}")
    print("-" * 65)
    
    for model in ['mbert', 'xlmr', 'mdeberta']:
        b = bin_results[model]
        f = fg_results[model]
        print(f"{model:<10} | {b['Same']:<7.2f} {b['Cross']:<7.2f} {b['Gap']:<7.2f} | {f['Same']:<7.2f} {f['Cross']:<7.2f} {f['Gap']:<7.2f}")

    # 3. Hidden Insights Check: JA/ZH as Source vs JA/ZH as Target
    print("\n" + "="*50)
    print("HIDDEN INSIGHTS: JA/ZH as Source vs Target (Fine-Grained MAE)")
    print("="*50)
    df_fg['is_source'] = df_fg.apply(lambda row: is_ja_zh_source(row['source_lang'], row['target_lang']), axis=1)
    df_fg['is_target'] = df_fg.apply(lambda row: is_ja_zh_target(row['source_lang'], row['target_lang']), axis=1)
    
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df_fg[df_fg['model'] == model]
        source_mae = compute_mae(m_df[m_df['is_source']])
        target_mae = compute_mae(m_df[m_df['is_target']])
        print(f"{model:<10}: JA/ZH as Source MAE = {source_mae:.2f} | JA/ZH as Target MAE = {target_mae:.2f} | Diff = {source_mae - target_mae:.2f}")

if __name__ == '__main__':
    main()
