#!/usr/bin/env python
"""
完整训练管道 - 统一输出版本

本脚本训练所有baseline模型和Bi-ARL,使用统一的输出目录结构。
所有输出文件整齐保存在 outputs/ 目录下。

训练序列:
    1. Bi-ARL (Bi-level Adversarial RL) - 100 episodes per seed
    2. Vanilla PPO - 100 episodes per seed  
    3. LSTM-IDS - 20 epochs per seed

Seeds: 从Config.SEEDS读取 (确保统计可靠性)
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 使用Config中的配置
from src.utils.config import Config

# 从Config读取配置
SEEDS = Config.SEEDS  # 使用Config中定义的seeds
EPISODES_RL = Config.RL_EPISODES  # 使用Config中定义的训练轮数
EPOCHS_LSTM = Config.LSTM_EPOCHS  # 使用Config中定义的LSTM周期

def print_header():
    print(f"\n{'='*70}")
    print(f"  完整训练管道 (使用Config配置)")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {Config.DATASET_NAME}")
    print(f"随机种子: {SEEDS} (from Config.SEEDS)")
    print(f"RL训练轮数: {EPISODES_RL} (from Config.RL_EPISODES)")
    print(f"LSTM训练周期: {EPOCHS_LSTM} (from Config.LSTM_EPOCHS)")
    print(f"\n输出目录: {Config.OUTPUT_DIR.absolute()}")
    print(f"  ├── models/   {Config.MODELS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"  ├── logs/     {Config.LOGS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"  ├── results/  {Config.RESULTS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"  └── checkpoints/ {Config.CHECKPOINTS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"{'='*70}\n")


def run_command(cmd, description):
    """运行命令并跟踪执行"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"命令: {' '.join(map(str, cmd))}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n成功 - {description}")
        print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        return True
    else:
        print(f"\n失败 - {description}")
        print(f"退出码: {result.returncode}")
        return False


def train_bilevel(seed):
    """训练 Bi-ARL"""
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'main_train_bilevel.py'),
        '--seed', str(seed),
        '--episodes', str(EPISODES_RL),
        '--dataset', Config.DATASET_NAME
    ]
    return run_command(cmd, f"Bi-ARL 训练 (Seed {seed})")


def train_vanilla_ppo(seed):
    """训练 Vanilla PPO"""
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'baselines' / 'vanilla_ppo.py'),
        '--seed', str(seed),
        '--episodes', str(EPISODES_RL),
        '--dataset', Config.DATASET_NAME
    ]
    return run_command(cmd, f"Vanilla PPO 训练 (Seed {seed})")


def train_lstm(seed):
    """训练 LSTM-IDS"""
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'baselines' / 'lstm_ids.py'),
        '--seed', str(seed),
        '--epochs', str(EPOCHS_LSTM),
        '--dataset', Config.DATASET_NAME
    ]
    return run_command(cmd, f"LSTM-IDS 训练 (Seed {seed})")


def train_fixed_attacker(seed):
    """训练 w/o Inner Loop 变体"""
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'baselines' / 'bilevel_fixed_attacker.py'),
        '--seed', str(seed),
        '--episodes', str(EPISODES_RL),
        '--dataset', Config.DATASET_NAME
    ]
    return run_command(cmd, f"Fixed Attacker 训练 (Seed {seed})")


def train_marl(seed):
    """训练 w/o Bi-level 变体"""
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'baselines' / 'marl_baseline.py'),
        '--seed', str(seed),
        '--episodes', str(EPISODES_RL),
        '--dataset', Config.DATASET_NAME
    ]
    return run_command(cmd, f"MARL 训练 (Seed {seed})")


def main():
    """主训练管道"""
    import argparse

    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--dataset", type=str, default="nsl-kdd", help="Dataset: nsl-kdd / unsw-nb15")
    args = parser.parse_args()
    Config.configure_dataset(args.dataset)
    print_header()

    results = []
    total_start = time.time()
    
    # 确保输出目录存在
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for seed in SEEDS:
        print(f"\n\n{'#'*70}")
        print(f"#  使用种子 {seed} 进行训练")
        print(f"{'#'*70}\n")
        
        # 1. 训练 Bi-ARL
        success = train_bilevel(seed)
        results.append((f"Bi-ARL (Seed {seed})", success))
        
        if not success:
            print(f"\n警告: Bi-ARL训练失败 (seed {seed}),继续下一个...")
        
        # 2. 训练 Vanilla PPO
        success = train_vanilla_ppo(seed)
        results.append((f"Vanilla PPO (Seed {seed})", success))
        
        if not success:
            print(f"\n警告: Vanilla PPO训练失败 (seed {seed}),继续下一个...")
        
        # 3. 训练 LSTM-IDS
        success = train_lstm(seed)
        results.append((f"LSTM-IDS (Seed {seed})", success))
        
        if not success:
            print(f"\n警告: LSTM-IDS训练失败 (seed {seed}),继续下一个...")

        # 4. 训练 Fixed Attacker
        success = train_fixed_attacker(seed)
        results.append((f"Fixed Attacker (Seed {seed})", success))

        if not success:
            print(f"\n警告: Fixed Attacker训练失败 (seed {seed}),继续下一个...")

        # 5. 训练 MARL
        success = train_marl(seed)
        results.append((f"MARL (Seed {seed})", success))

        if not success:
            print(f"\n警告: MARL训练失败 (seed {seed}),继续下一个...")
    
    # 打印最终总结
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  训练管道完成")
    print(f"{'='*70}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n{'='*70}")
    print(f"  结果汇总")
    print(f"{'='*70}")
    for name, success in results:
        status = "成功" if success else "失败"
        print(f"{name:30} {status}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"{'='*70}")
    print(f"成功率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    print(f"\n所有输出已保存至: {Config.OUTPUT_DIR.absolute()}")
    print(f"{'='*70}\n")
    
    print(f"\n下一步操作:")
    print(f"1. 整理输出: python scripts/organize_outputs.py")
    print(f"2. 运行评估: python src/experiments.py --seed 42")
    print(f"3. 查看模型: {Config.MODELS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"4. 查看日志: {Config.LOGS_DIR.relative_to(Config.OUTPUT_DIR)}")
    print(f"5. 查看结果: {Config.RESULTS_DIR.relative_to(Config.OUTPUT_DIR)}")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n警告: 训练被用户中断 (Ctrl+C)")
        print("部分模型可能已保存至 outputs/ 目录")
        sys.exit(1)
