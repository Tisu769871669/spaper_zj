#!/usr/bin/env python
"""
使用标准对抗攻击(FGSM/PGD)评估模型鲁棒性

替换之前的高斯噪声评估,提升科学严谨性
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.data_loader import build_data_loader
from src.agents.defender_agent import DefenderAgent
from src.attacks.fgsm import FGSMAttack, PGDAttack
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def action_to_prediction(action: int) -> int:
    return 1 if action >= 5 else 0


def evaluate_under_attack(model, X_test, y_test, attack_name='FGSM', epsilon=0.3):
    """
    在对抗攻击下评估模型
    
    参数:
        model: Defender模型
        X_test: 测试数据
        y_test: 测试标签
        attack_name: 'FGSM' 或 'PGD'
        epsilon: 扰动幅度
    
    返回:
        dict: 性能指标
    """
    print(f"\n[{attack_name}攻击评估, ε={epsilon}]")
    
    # 生成对抗样本
    if attack_name == 'FGSM':
        attack = FGSMAttack(model, epsilon=epsilon, device=Config.DEVICE)
        X_adv = attack.generate_batch(X_test, y_test)
    elif attack_name == 'PGD':
        attack = PGDAttack(model, epsilon=epsilon, device=Config.DEVICE)
        X_adv = attack.generate_batch(X_test, y_test)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
    
    # 在对抗样本上评估
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_adv)):
            state = torch.FloatTensor(X_adv[i]).unsqueeze(0).to(Config.DEVICE)
            action_probs = model(state)
            action = torch.argmax(action_probs, dim=1).item()
            pred = action_to_prediction(action)
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 计算指标
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    fpr = np.sum((predictions == 1) & (y_test == 0)) / np.sum(y_test == 0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    
    return {
        'Attack': attack_name,
        'Epsilon': epsilon,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': prec,
        'F1': f1,
        'FPR': fpr
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nsl-kdd", help="Dataset: nsl-kdd / unsw-nb15")
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)

    print("\n" + "="*70)
    print("  标准对抗攻击鲁棒性评估 (FGSM-style & PGD-style)")
    print("="*70 + "\n")
    
    # 加载数据
    print("加载NSL-KDD测试数据...")
    loader = build_data_loader(Config.DATASET_NAME)
    loader.load_data(mode='train')
    X_test, y_test = loader.load_data(mode='test')
    
    print(f"测试集大小: {len(X_test)}")
    
    model_specs = {
        'Bi-ARL': ("BiARL", "defender"),
        'Vanilla PPO': ("VanillaPPO", "model"),
        'MARL': ("MARL", "defender"),
    }
    
    # 简化攻击:基于数值扰动(类FGSM效果)
    attack_configs = [
        ('Clean', None, 0.0),
        ('FGSM-ε0.1', 'FGSM', 0.1),
        ('FGSM-ε0.3', 'FGSM', 0.3),
    ]
    
    results = []
    
    for model_name, (model_type, model_file) in model_specs.items():
        for seed in Config.SEEDS:
            model_path = Config.find_model_file(model_type, seed, model_file)
            if not model_path.exists():
                print(f"\n跳过 {model_name} seed={seed}: 模型不存在")
                continue

            print(f"\n{'='*70}")
            print(f"  评估: {model_name} (seed={seed})")
            print(f"{'='*70}")

            defender = DefenderAgent().to(Config.DEVICE)
            defender.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            defender.eval()

            for attack_name, attack_type, epsilon in attack_configs:
                print(f"\n[{attack_name}]")

                if attack_type is None:
                    predictions = []
                    with torch.no_grad():
                        for i in range(len(X_test)):
                            state = torch.FloatTensor(X_test[i]).unsqueeze(0).to(Config.DEVICE)
                            action_probs = defender(state)
                            action = torch.argmax(action_probs, dim=1).item()
                            predictions.append(action_to_prediction(action))

                    predictions = np.array(predictions)
                    acc = accuracy_score(y_test, predictions)
                    recall = recall_score(y_test, predictions)
                    prec = precision_score(y_test, predictions, zero_division=0)
                    f1 = f1_score(y_test, predictions, zero_division=0)
                    fpr = np.sum((predictions == 1) & (y_test == 0)) / np.sum(y_test == 0)
                    metrics = {
                        'Attack': attack_name,
                        'Epsilon': epsilon,
                        'Accuracy': acc,
                        'Recall': recall,
                        'Precision': prec,
                        'F1': f1,
                        'FPR': fpr
                    }
                else:
                    metrics = evaluate_under_attack(defender, X_test, y_test, attack_name=attack_type, epsilon=epsilon)
                    metrics['Attack'] = attack_name

                print(f"  Recall: {metrics['Recall']:.4f}, Precision: {metrics['Precision']:.4f}, FPR: {metrics['FPR']:.4f}")

                results.append({
                    'Model': model_name,
                    'Seed': seed,
                    **metrics
                })
    
    # 保存结果
    df = pd.DataFrame(results)
    suffix = Config.DATASET_NAME.replace("-", "_")
    detailed_output = Config.RESULTS_DIR / f"adversarial_robustness_detailed_{suffix}.csv"
    summary_output = Config.RESULTS_DIR / f"adversarial_robustness_{suffix}.csv"
    df.to_csv(detailed_output, index=False)

    summary_df = df.groupby(['Model', 'Attack', 'Epsilon'])[['Accuracy', 'Recall', 'Precision', 'F1', 'FPR']].mean().reset_index()
    summary_df.to_csv(summary_output, index=False)
    
    print(f"\n结果已保存: {summary_output}")
    print(f"明细已保存: {detailed_output}")
    
    # 生成对比表
    print("\n" + "="*70)
    print("  鲁棒性对比")
    print("="*70 + "\n")
    
    print(f"{'Model':<20} {'Attack':<15} {'Recall':>10} {'FPR':>10}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<20} {row['Attack']:<15} {row['Recall']:>10.4f} {row['FPR']:>10.4f}")
    
    print("\n" + "="*70)
    print("  完成标准对抗攻击评估!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
