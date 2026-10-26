import pandas as pd

print("="*60)
print("DEEP VERIFICATION OF TABLE III")
print("="*60)

def print_stats(name, file_path, metric, higher_is_better):
    df = pd.read_csv(file_path)
    
    # Manually determine supervised vs zero-shot to avoid trusting 'is_supervised' column
    df['is_sup'] = df['source_lang'] == df['target_lang']
    
    print(f"\n--- {name} ---")
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df[df['model'] == model]
        
        # Calculate pair-level metrics first (like the actual paper evaluation script would)
        # This is important! The mean of (all rows) vs mean of (pair means) can be slightly different
        # if the number of rows per pair varies.
        
        def calc_metric(group):
            if metric == 'acc':
                return group['correct'].mean() * 100
            else: # MAE
                return (group['true_label'] - group['pred_label']).abs().mean() * 100
        
        pair_metrics = m_df.groupby(['source_lang', 'target_lang', 'is_sup']).apply(calc_metric).reset_index(name='val')
        
        zs_pairs = pair_metrics[pair_metrics['is_sup'] == False]
        sup_pairs = pair_metrics[pair_metrics['is_sup'] == True]
        
        zs_mean = zs_pairs['val'].mean()
        sup_mean = sup_pairs['val'].mean()
        
        print(f"{model:<10} | Zero-Shot {metric.upper()}: {zs_mean:>6.2f} (n={len(zs_pairs)} pairs) | Sup {metric.upper()}: {sup_mean:>6.2f} (n={len(sup_pairs)} pairs)")
        
print_stats('FINE-GRAINED (MAE, Lower is better)', 'results/confidence_data.csv', 'mae', False)
print_stats('BINARIZED (Accuracy, Higher is better)', 'results/confidence_data_binarized.csv', 'acc', True)
print_stats('TRI-CLASS (MAE, Lower is better)', 'results/confidence_data_triclass.csv', 'mae', False)
