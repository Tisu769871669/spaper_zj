
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import Config
from src.utils.data_loader import NSLKDDLoader
from src.agents.defender_agent import DefenderAgent
from src.agents.attacker_agent import AttackerAgent

def calculate_metrics(y_true, y_pred, label_prefix=""):
    """
    计算全面指标: Acc, Recall, Precision, F1, FPR
    """
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"{label_prefix}: Acc={acc:.4f}, Recall={rec:.4f}, Prec={prec:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
    
    return {
        "Acc": acc, "Recall": rec, "Precision": prec, "F1": f1, "FPR": fpr
    }

def apply_smart_noise(X, noise_sigma=0.5):
    """
    Apply Gaussian noise ONLY to continuous features.
    Skip categorical features at indices [1, 2, 3] (protocol, service, flag).
    """
    # 假设 X 是 numpy array
    noise = np.random.normal(0, noise_sigma, size=X.shape)
    
    # Mask categorical indices
    # indices 1, 2, 3 corresponds to protocol_type, service, flag
    mask = np.ones(X.shape[1], dtype=bool)
    mask[[1, 2, 3]] = False 
    
    # 只在 mask=True 的列加噪声
    X_noisy = X.copy()
    X_noisy[:, mask] += noise[:, mask]
    
    return np.clip(X_noisy, 0, 1)

def generate_adversarial_data(X_test, attacker_model):
    """
    使用训练好的 Attacker 生成对抗样本。
    """
    attacker_model.eval()
    X_adv = X_test.copy()
    
    with torch.no_grad():
        for i in range(len(X_test)):
            state = X_test[i]
            # Attacker 决定扰动动作
            action, _ = attacker_model.get_action(state)
            
            # Action 0-9: 影响不同的特征组
            start_idx = (action % 10) * 4 
            end_idx = min(start_idx + 4, len(state))
            
            # 智能扰动: 跳过 categorical indices [1, 2, 3]
            # 这里简化处理：如果是 categorical 就不加，或者用 smart noise 逻辑
            # 为了效率，我们暂时在 loop 里只对非 categorical 加
            target_indices = np.arange(start_idx, end_idx)
            valid_indices = [idx for idx in target_indices if idx not in [1, 2, 3]]
            
            if valid_indices:
                # 加大攻击力度 (Simulate Strong Adversarial Attack)
                noise = np.random.normal(0, 0.8, size=len(valid_indices))
                X_adv[i][valid_indices] += noise
            
    # Clip to valid range
    X_adv = np.clip(X_adv, 0, 1)
    return X_adv

def evaluate_rf_baseline(X_train, y_train, X_test, y_test, X_adv=None):
    """
    基准模型: 随机森林 (支持对抗数据测试)
    """
    print("\n--- Running Baseline 1: Random Forest ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    results = {}
    
    # 1. Clean
    y_pred = clf.predict(X_test)
    results['Clean'] = calculate_metrics(y_test, y_pred, "Random Forest [Clean]")
    
    # 2. Adversarial (Fairness Check)
    if X_adv is not None:
        y_pred_adv = clf.predict(X_adv)
        results['Adv'] = calculate_metrics(y_test, y_pred_adv, "Random Forest [Advrd]")
    
    # 3. Universal Noise Attack (Stress Test)
    print("--- Running Stress Test: Universal Noise (Sigma=0.5) ---")
    X_stress = apply_smart_noise(X_test, noise_sigma=0.5)
    
    # Evaluate RF on Stress
    y_pred_rf_stress = clf.predict(X_stress)
    results['Stress'] = calculate_metrics(y_test, y_pred_rf_stress, "Random Forest [Stress]")
    
    return results

def evaluate_rl_agent(X_test, y_test, X_adv=None, seed=42):
    """
    Ours: Bi-ARL Defender
    
    参数:
        seed: 随机种子,用于找到对应的模型文件
    """
    print(f"\n--- Running Ours: Bi-ARL Defender (Seed {seed}) ---")
    defender = DefenderAgent().to(Config.DEVICE)
    
    # 更新路径:先尝试从 outputs/ 目录加载,如果不存在则尝试 src/
    model_path_new = os.path.join(project_root, "outputs", "models", "BiARL", f"seed{seed}", "defender.pth")
    model_path_old = os.path.join(project_root, "src", f"defender_bilevel_seed{seed}.pth")
    model_path_legacy = os.path.join(project_root, "src", "defender.pth")
    
    # 按优先级尝试加载
    for path in [model_path_new, model_path_old, model_path_legacy]:
        if os.path.exists(path):
            print(f"Loading model from: {os.path.relpath(path, project_root)}")
            model_path = path
            break
    else:
        print("Model not found! Tried:")
        print(f"  1. {os.path.relpath(model_path_new, project_root)}")
        print(f"  2. {os.path.relpath(model_path_old, project_root)}")
        print(f"  3. {os.path.relpath(model_path_legacy, project_root)}")
        return None
        
    defender.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    defender.eval()
    
    def predict(model, data):
        preds = []
        with torch.no_grad():
            for i in range(len(data)):
                action, _ = model.get_action(data[i])
                preds.append(0 if action < 5 else 1)
        return np.array(preds)
    
    results = {}

    # 1. Clean
    y_pred = predict(defender, X_test)
    results['Clean'] = calculate_metrics(y_test, y_pred, "Bi-ARL Defender [Clean]")

    # 2. Adversarial
    if X_adv is not None:
        y_pred_adv = predict(defender, X_adv)
        results['Adv'] = calculate_metrics(y_test, y_pred_adv, "Bi-ARL Defender [Advrd]")
        
    # 3. Universal Stress Test
    X_stress = apply_smart_noise(X_test, noise_sigma=0.5)
    y_pred_stress = predict(defender, X_stress)
    results['Stress'] = calculate_metrics(y_test, y_pred_stress, "Bi-ARL Defender [Stress]")
        
    return results

def save_results(rf_res, rl_res, seed):
    """保存单次运行结果到 CSV 以便后续统计"""
    data = []
    
    # Helper to flatten
    def add_row(model_name, condition, metrics):
        row = {'Model': model_name, 'Condition': condition, 'Seed': seed}
        row.update(metrics)
        data.append(row)
        
    for cond, met in rf_res.items():
        add_row('RandomForest', cond, met)
        
    if rl_res:
        for cond, met in rl_res.items():
            add_row('Bi-ARL', cond, met)
            
    df = pd.DataFrame(data)
    save_file = os.path.join(project_root, "src", "experiment_results.csv")
    # Append if exists
    if os.path.exists(save_file):
        df.to_csv(save_file, mode='a', header=False, index=False)
    else:
        df.to_csv(save_file, index=False)
    print(f"Results appended to {save_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args = parser.parse_args()
    
    Config.set_seed(args.seed)
    loader = NSLKDDLoader()
    print(f"Loading Data (Seed {args.seed})...")
    X_train, y_train = loader.load_data(mode='train')
    X_test, y_test = loader.load_data(mode='test')
    
    # Load Attacker to generate Shared Adversarial Set
    attacker = AttackerAgent().to(Config.DEVICE)
    att_path = os.path.join(project_root, "src", "attacker.pth")
    X_adv = None
    if os.path.exists(att_path):
        print("Generating Adversarial Samples for Fairness Comparison...")
        attacker.load_state_dict(torch.load(att_path, map_location=Config.DEVICE))
        X_adv = generate_adversarial_data(X_test, attacker)

    # 1. Run Baseline
    rf_results = evaluate_rf_baseline(X_train, y_train, X_test, y_test, X_adv)
    
    # 2. Run RL Agent
    rl_results = evaluate_rl_agent(X_test, y_test, X_adv, seed=args.seed)
    
    # 3. Save
    save_results(rf_results, rl_results, args.seed)

if __name__ == "__main__":
    main()
