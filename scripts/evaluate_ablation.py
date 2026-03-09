#!/usr/bin/env python
"""
Ablation Study 统一评估脚本

对比4个变体:
1. Full Bi-ARL
2. w/o Inner Loop (Fixed Attacker)
3. w/o Attacker (Vanilla PPO)
4. w/o Bi-level (MARL)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_loader import build_data_loader
from src.agents.defender_agent import DefenderAgent
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import warnings
warnings.filterwarnings('ignore')


def action_to_prediction(action: int) -> int:
    return 1 if action >= 5 else 0


def evaluate_defender(model_path, test_X, test_y, test_X_stress, test_y_stress, model_name="Model", seed=None):
    """评估单个Defender模型"""
    
    print(f"\n{'='*70}")
    suffix = f" (seed={seed})" if seed is not None else ""
    print(f"  评估: {model_name}{suffix}")
    print(f"{'='*70}")
    
    if not model_path.exists():
        print(f"[Skip] 模型不存在: {model_path}")
        return None
    
    # 加载模型
    defender = DefenderAgent().to(Config.DEVICE)
    defender.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    defender.eval()
    
    results = {}
    
    # Clean测试
    print(f"\n[Clean测试]")
    predictions = []
    with torch.no_grad():
        for i in range(len(test_X)):
            state = torch.FloatTensor(test_X[i]).unsqueeze(0).to(Config.DEVICE)
            action_probs = defender(state)
            action = torch.argmax(action_probs, dim=1).item()
            pred = action_to_prediction(action)
            predictions.append(pred)
    
    predictions = np.array(predictions)
    acc = accuracy_score(test_y, predictions)
    recall = recall_score(test_y, predictions)
    prec = precision_score(test_y, predictions, zero_division=0)
    f1 = f1_score(test_y, predictions, zero_division=0)
    fpr = np.sum((predictions == 1) & (test_y == 0)) / np.sum(test_y == 0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    
    results['Clean'] = {
        'Accuracy': acc,
        'Recall': recall,
        'Precision': prec,
        'F1': f1,
        'FPR': fpr
    }
    
    # Stress测试
    print(f"\n[Stress测试]")
    predictions_stress = []
    with torch.no_grad():
        for i in range(len(test_X_stress)):
            state = torch.FloatTensor(test_X_stress[i]).unsqueeze(0).to(Config.DEVICE)
            action_probs = defender(state)
            action = torch.argmax(action_probs, dim=1).item()
            pred = action_to_prediction(action)
            predictions_stress.append(pred)
    
    predictions_stress = np.array(predictions_stress)
    acc_s = accuracy_score(test_y_stress, predictions_stress)
    recall_s = recall_score(test_y_stress, predictions_stress)
    prec_s = precision_score(test_y_stress, predictions_stress, zero_division=0)
    f1_s = f1_score(test_y_stress, predictions_stress, zero_division=0)
    fpr_s = np.sum((predictions_stress == 1) & (test_y_stress == 0)) / np.sum(test_y_stress == 0)
    
    print(f"  Accuracy:  {acc_s:.4f}")
    print(f"  Recall:    {recall_s:.4f}")
    print(f"  Precision: {prec_s:.4f}")
    print(f"  F1:        {f1_s:.4f}")
    print(f"  FPR:       {fpr_s:.4f}")
    
    results['Stress'] = {
        'Accuracy': acc_s,
        'Recall': recall_s,
        'Precision': prec_s,
        'F1': f1_s,
        'FPR': fpr_s
    }
    
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsl-kdd", help="Dataset: nsl-kdd / unsw-nb15")
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)

    print("\n" + "="*70)
    print("  Ablation Study - 4变体性能对比")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载测试数据...")
    loader = build_data_loader(Config.DATASET_NAME)
    loader.load_data(mode='train')
    test_X, test_y = loader.load_data(mode='test')
    
    # 生成Stress测试数据
    print("生成Stress测试数据 (σ=0.5)...")
    noise = np.random.normal(0, 0.5, test_X.shape)
    mask = np.ones(test_X.shape[1], dtype=bool)
    mask[[1, 2, 3]] = False
    test_X_stress = test_X.copy()
    test_X_stress[:, mask] = np.clip(test_X_stress[:, mask] + noise[:, mask], 0, 1)
    test_y_stress = test_y.copy()
    
    variant_specs = {
        'Full Bi-ARL': ("BiARL", "defender"),
        'w/o Inner Loop': ("FixedAttacker", "defender"),
        'w/o Attacker (Vanilla PPO)': ("VanillaPPO", "model"),
        'w/o Bi-level (MARL)': ("MARL", "defender"),
    }

    detailed_rows = []
    all_results = {}
    for name, (model_type, model_file) in variant_specs.items():
        all_results[name] = {}
        for seed in Config.SEEDS:
            model_path = Config.find_model_file(model_type, seed, model_file)
            results = evaluate_defender(model_path, test_X, test_y, test_X_stress, test_y_stress, name, seed=seed)
            if not results:
                continue

            for condition in ['Clean', 'Stress']:
                row = {'Variant': name, 'Seed': seed, 'Condition': condition}
                row.update(results[condition])
                detailed_rows.append(row)

        variant_df = pd.DataFrame([row for row in detailed_rows if row['Variant'] == name])
        for condition in ['Clean', 'Stress']:
            cond_df = variant_df[variant_df['Condition'] == condition]
            if cond_df.empty:
                continue
            all_results[name][condition] = {
                'Accuracy': cond_df['Accuracy'].mean(),
                'Recall': cond_df['Recall'].mean(),
                'Precision': cond_df['Precision'].mean(),
                'F1': cond_df['F1'].mean(),
                'FPR': cond_df['FPR'].mean(),
            }
    
    # 生成对比表
    print("\n" + "="*70)
    print("  结果汇总")
    print("="*70 + "\n")
    
    # Clean条件对比
    print("Clean 条件:")
    print(f"{'Variant':<30} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8}")
    print("-" * 70)
    for name, results in all_results.items():
        if 'Clean' in results:
            r = results['Clean']
            print(f"{name:<30} {r['Recall']:>8.4f} {r['Precision']:>10.4f} {r['F1']:>8.4f} {r['FPR']:>8.4f}")
    
    print("\nStress 条件:")
    print(f"{'Variant':<30} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FPR':>8}")
    print("-" * 70)
    for name, results in all_results.items():
        if 'Stress' in results:
            r = results['Stress']
            print(f"{name:<30} {r['Recall']:>8.4f} {r['Precision']:>10.4f} {r['F1']:>8.4f} {r['FPR']:>8.4f}")
    
    # 生成LaTeX表格
    print("\n" + "="*70)
    print("  LaTeX表格代码")
    print("="*70 + "\n")
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Study: Component-wise Performance Comparison}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Variant & Recall (Clean) & Recall (Stress) & FPR (Clean) & FPR (Stress) \\\\")
    print("\\midrule")
    
    for name, results in all_results.items():
        clean = results.get('Clean', {})
        stress = results.get('Stress', {})
        
        # 简化名称
        short_name = name.replace("w/o ", "").replace(" (", " \\\\textit{(").replace(")", ")}")
        
        print(f"{short_name} & {clean.get('Recall', 0):.3f} & "
              f"{stress.get('Recall', 0):.3f} & "
              f"{clean.get('FPR', 0):.3f} & {stress.get('FPR', 0):.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # 保存结果
    suffix = Config.DATASET_NAME.replace("-", "_")
    detailed_output = Config.RESULTS_DIR / f"ablation_results_detailed_{suffix}.csv"
    summary_output = Config.RESULTS_DIR / f"ablation_results_{suffix}.csv"

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(detailed_output, index=False)

    summary_rows = []
    for name, results in all_results.items():
        for condition in ['Clean', 'Stress']:
            if condition in results:
                row = {'Variant': name, 'Condition': condition}
                row.update(results[condition])
                summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(summary_output, index=False)
    print(f"\n结果已保存: {summary_output}")
    print(f"明细已保存: {detailed_output}")
    
    print("\n" + "="*70)
    print("  Ablation Study完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
