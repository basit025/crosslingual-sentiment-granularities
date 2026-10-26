import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

def calc_pair_metrics(file_path, metric, norm_factor):
    df = pd.read_csv(file_path)
    df['is_sup'] = df['source_lang'] == df['target_lang']
    
    results = {}
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df[df['model'] == model]
        
        def calc_metric(group):
            if metric == 'acc':
                val = (1.0 - group['correct'].mean()) * 100 # Error %
            else: # MAE
                val = (group['true_label'] - group['pred_label']).abs().mean() * 100
            return val / norm_factor
        
        # Calculate per-pair, then average
        pair_metrics = m_df.groupby(['source_lang', 'target_lang', 'is_sup']).apply(calc_metric).reset_index(name='val')
        
        zs_mean = pair_metrics[pair_metrics['is_sup'] == False]['val'].mean()
        sup_mean = pair_metrics[pair_metrics['is_sup'] == True]['val'].mean()
        
        results[model] = {
            'Sup': sup_mean,
            'ZS': zs_mean,
            'Gap': zs_mean - sup_mean
        }
    return results

def main():
    os.makedirs('IMPROVEMENTS/phase_2_3', exist_ok=True)
    
    print("Normalizing Metrics...")
    bin_results = calc_pair_metrics('results/confidence_data_binarized.csv', 'acc', 1.0)
    tri_results = calc_pair_metrics('results/confidence_data_triclass.csv', 'mae', 2.0)
    fg_results = calc_pair_metrics('results/confidence_data.csv', 'mae', 4.0)
    
    print("="*60)
    print("NORMALIZED SUPERVISED-VS-ZERO-SHOT GAP (0-100 Scale)")
    print("="*60)
    print(f"{'Model':<10} | {'Granularity':<12} | {'Sup Error':<9} | {'ZS Error':<9} | {'Norm Gap':<9}")
    print("-" * 60)
    
    for model in ['mbert', 'xlmr', 'mdeberta']:
        b = bin_results[model]
        t = tri_results[model]
        f = fg_results[model]
        
        print(f"{model.upper():<10} | {'Binarized':<12} | {b['Sup']:>9.2f} | {b['ZS']:>9.2f} | {b['Gap']:>9.2f}")
        print(f"{'':<10} | {'Tri-Class':<12} | {t['Sup']:>9.2f} | {t['ZS']:>9.2f} | {t['Gap']:>9.2f}")
        print(f"{'':<10} | {'Fine-Grained':<12} | {f['Sup']:>9.2f} | {f['ZS']:>9.2f} | {f['Gap']:>9.2f}")
        print("-" * 60)

if __name__ == '__main__':
    main()
