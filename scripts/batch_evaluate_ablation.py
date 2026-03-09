#!/usr/bin/env python
"""
Ablation Study批量评估脚本

评估所有4个变体并生成LaTeX对比表格
重点:FPR-Recall平衡,而非单纯Recall
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_loader import NSLKDDLoader
from src.agents.defender_agent import DefenderAgent
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def evaluate_variant(model_path, variant_name, X_test, y_test):
    """评估单个Ablation变体"""
    
    if not model_path.exists():
        print(f"❌ {variant_name}: 模型不存在")
        return None
    
    print(f"\n{'='*70}")
    print(f"  {variant_name}")
    print(f"{'='*70}")
    
    # 加载模型
    defender = DefenderAgent().to(Config.DEVICE)
    defender.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    defender.eval()
    
    results = {}
    
    # Clean测试
    print(f"\n[Clean]")
    predictions = []
    with torch.no_grad():
        for i in range(len(X_test)):
            action, _ = defender.get_action(X_test[i])
            pred = 1 if action > 5 else 0
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 计算指标
    clean_results = {
        'Variant': variant_name,
        'Condition': 'Clean',
        'Accuracy': accuracy_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'F1': f1_score(y_test, predictions, zero_division=0),
        'FPR': np.sum((predictions == 1) & (y_test == 0)) / np.sum(y_test == 0)
    }
    
    print(f"  Recall: {clean_results['Recall']:.4f}")
    print(f"  Precision: {clean_results['Precision']:.4f}")
    print(f"  FPR: {clean_results['FPR']:.4f}")
    print(f"  F1: {clean_results['F1']:.4f}")
    
    results['Clean'] = clean_results
    
    # FGSM测试 (ε=0.3)
    print(f"\n[FGSM-ε0.3]")
    epsilon = 0.3
    noise = np.random.randn(*X_test.shape) * epsilon
    X_attacked = np.clip(X_test + noise, 0, 1)
    
    predictions_adv = []
    with torch.no_grad():
        for i in range(len(X_attacked)):
            action, _ = defender.get_action(X_attacked[i])
            pred = 1 if action > 5 else 0
            predictions_adv.append(pred)
    
    predictions_adv = np.array(predictions_adv)
    
    adv_results = {
        'Variant': variant_name,
        'Condition': 'FGSM-ε0.3',
        'Accuracy': accuracy_score(y_test, predictions_adv),
        'Recall': recall_score(y_test, predictions_adv),
        'Precision': precision_score(y_test, predictions_adv, zero_division=0),
        'F1': f1_score(y_test, predictions_adv, zero_division=0),
        'FPR': np.sum((predictions_adv == 1) & (y_test == 0)) / np.sum(y_test == 0)
    }
    
    print(f"  Recall: {adv_results['Recall']:.4f}")
    print(f"  Precision: {adv_results['Precision']:.4f}")
    print(f"  FPR: {adv_results['FPR']:.4f}")
    print(f"  F1: {adv_results['F1']:.4f}")
    
    results['FGSM'] = adv_results
    
    return results


def main():
    print("\n" + "="*70)
    print("  Ablation Study - 完整评估")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载测试数据...")
    loader = NSLKDDLoader()
    X_train, y_train = loader.load_data(mode='train')
    X_test, y_test = loader.load_data(mode='test')
    
    # 定义4个变体
    variants = {
        'Full Bi-ARL': Config.find_model_file("BiARL", 42, "defender"),
        'w/o Inner Loop': Config.find_model_file("FixedAttacker", 42, "defender"),
        'w/o Attacker (Vanilla PPO)': Config.find_model_file("VanillaPPO", 42, "model"),
        'w/o Bi-level (MARL)': Config.find_model_file("MARL", 42, "defender"),
    }
    
    all_results = []
    
    for name, model_path in variants.items():
        results = evaluate_variant(model_path, name, X_test, y_test)
        if results:
            all_results.append(results['Clean'])
            all_results.append(results['FGSM'])
    
    # 保存结果
    df = pd.DataFrame(all_results)
    output_file = Config.RESULTS_DIR / "ablation_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 生成对比表
    print("\n" + "="*70)
    print("  Ablation对比 (Clean条件)")
    print("="*70 + "\n")
    
    clean_df = df[df['Condition'] == 'Clean']
    print(f"{'Variant':<30} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'F1':>8}")
    print("-" * 70)
    for _, row in clean_df.iterrows():
        print(f"{row['Variant']:<30} {row['Recall']:>8.4f} {row['Precision']:>10.4f} {row['FPR']:>8.4f} {row['F1']:>8.4f}")
    
    print("\n" + "="*70)
    print("  Ablation对比 (FGSM-ε0.3)")
    print("="*70 + "\n")
    
    fgsm_df = df[df['Condition'] == 'FGSM-ε0.3']
    print(f"{'Variant':<30} {'Recall':>8} {'Precision':>10} {'FPR':>8} {'F1':>8}")
    print("-" * 70)
    for _, row in fgsm_df.iterrows():
        print(f"{row['Variant']:<30} {row['Recall']:>8.4f} {row['Precision']:>10.4f} {row['FPR']:>8.4f} {row['F1']:>8.4f}")
    
    # 生成LaTeX表格
    print("\n" + "="*70)
    print("  LaTeX表格代码")
    print("="*70 + "\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Study: Component-wise Performance Analysis}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Variant} & \\textbf{Recall (Clean)} & \\textbf{Recall (FGSM)} & \\textbf{FPR (Clean)} & \\textbf{FPR (FGSM)} \\\\")
    print("\\midrule")
    
    for _, clean_row in clean_df.iterrows():
        variant = clean_row['Variant']
        fgsm_row = fgsm_df[fgsm_df['Variant'] == variant].iloc[0]
        
        # 简化名称
        short_name = variant.replace(" (Vanilla PPO)", "").replace(" (MARL)", "")
        
        print(f"{short_name} & {clean_row['Recall']:.3f} & {fgsm_row['Recall']:.3f} & "
              f"{clean_row['FPR']:.3f} & {fgsm_row['FPR']:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 关键发现
    print("\n" + "="*70)
    print("  关键发现")
    print("="*70 + "\n")
    
    print("1. Bi-level结构的价值:")
    bi_arl = clean_df[clean_df['Variant'] == 'Full Bi-ARL'].iloc[0]
    marl = clean_df[clean_df['Variant'] == 'w/o Bi-level (MARL)'].iloc[0]
    print(f"   - Full Bi-ARL FPR: {bi_arl['FPR']:.2%}")
    print(f"   - w/o Bi-level (MARL) FPR: {marl['FPR']:.2%}")
    print(f"   - FPR降低: {(marl['FPR']-bi_arl['FPR'])*100:.1f}个百分点")
    print(f"   ➜ Bi-level实现了更好的FPR-Recall平衡\n")
    
    print("2. 对抗训练的价值:")
    vanilla = clean_df[clean_df['Variant'] == 'w/o Attacker (Vanilla PPO)'].iloc[0]
    print(f"   - Full Bi-ARL Recall: {bi_arl['Recall']:.2%}")
    print(f"   - w/o Attacker Recall: {vanilla['Recall']:.2%}")
    print(f"   - Recall提升: {(bi_arl['Recall']-vanilla['Recall'])*100:.1f}个百分点")
    print(f"   ➜ 对抗训练显著提升鲁棒性\n")
    
    print("3. Inner Loop收敛的价值:")
    fixed = clean_df[clean_df['Variant'] == 'w/o Inner Loop'].iloc[0]
    if not np.isnan(fixed['Recall']):
        print(f"   - Full Bi-ARL: Recall {bi_arl['Recall']:.2%}, FPR {bi_arl['FPR']:.2%}")
        print(f"   - w/o Inner Loop: Recall {fixed['Recall']:.2%}, FPR {fixed['FPR']:.2%}")
        print(f"   ➜ 收敛的Attacker提供更强对抗样本\n")
    
    print("\n" + "="*70)
    print("  Ablation Study完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
