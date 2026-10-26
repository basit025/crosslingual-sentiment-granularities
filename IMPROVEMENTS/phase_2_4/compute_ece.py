import pandas as pd
import numpy as np
import os

def expected_calibration_error(y_true, y_pred, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = y_pred == y_true
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item() * 100 # return as percentage

def main():
    os.makedirs('IMPROVEMENTS/phase_2_4', exist_ok=True)
    df = pd.read_csv('results/confidence_data.csv')
    df['is_sup'] = df['source_lang'] == df['target_lang']
    
    results = {}
    for model in ['mbert', 'xlmr', 'mdeberta']:
        m_df = df[df['model'] == model]
        
        sup_df = m_df[m_df['is_sup'] == True]
        zs_df = m_df[m_df['is_sup'] == False]
        
        sup_ece = expected_calibration_error(sup_df['true_label'], sup_df['pred_label'], sup_df['confidence'])
        zs_ece = expected_calibration_error(zs_df['true_label'], zs_df['pred_label'], zs_df['confidence'])
        
        results[model] = {
            'Sup': sup_ece,
            'ZS': zs_ece,
            'Gap': zs_ece - sup_ece
        }
        
    print("="*50)
    print("EXPECTED CALIBRATION ERROR (ECE %)")
    print("="*50)
    print(f"{'Model':<10} | {'Supervised':<10} | {'Zero-Shot':<10} | {'Gap':<10}")
    print("-" * 47)
    for model in ['mbert', 'xlmr', 'mdeberta']:
        r = results[model]
        print(f"{model.upper():<10} | {r['Sup']:<10.2f} | {r['ZS']:<10.2f} | {r['Gap']:<10.2f}")

if __name__ == '__main__':
    main()
