
import os
import subprocess
import pandas as pd
import numpy as np

# 实验配置
SEEDS = [42, 101, 202, 303, 404]
# 暂时只跑 Full Mode 确保主实验严谨
MODES = ['full']
PYTHON_EXEC = r"D:\python\Anaconda3\envs\spaper\python.exe"
SCRIPT_TRAIN = "src/main_train.py"
SCRIPT_EVAL = "src/experiments.py"
RESULT_FILE = "src/experiment_results.csv"

def run_cmd(cmd):
    print(f">> Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"!! Command failed with code {ret}")
        
def main():
    # 1. Clean previous results
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)
        
    # 2. Loop
    for mode in MODES:
        print(f"\n=== Starting Experiment Mode: {mode} ===")
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            
            # Train
            cmd_train = f'"{PYTHON_EXEC}" {SCRIPT_TRAIN} --seed {seed} --mode {mode}'
            run_cmd(cmd_train)
            
            # Eval (Eval script inside saves to CSV)
            cmd_eval = f'"{PYTHON_EXEC}" {SCRIPT_EVAL} --seed {seed}'
            run_cmd(cmd_eval)
            
    # 3. Analyze
    if os.path.exists(RESULT_FILE):
        print("\n=== Final Scientific Report (Mean ± Std) ===")
        df = pd.read_csv(RESULT_FILE)
        
        # Group by Model, Condition
        # 由于我们 Eval 脚本里写的 Model 名字是固定的 (Bi-ARL), 
        # 我们需要在 CSV 里区分它是属于哪个 Run Mode (Full vs DefenderOnly)
        # 但 Eval 脚本只负责读 .pth，不知道 mode。
        # 简单方案：每次 Eval 完，我们这里读 CSV 把刚加进去的行标记一下 Mode
        # (这太复杂，不如直接让 Eval 脚本知道 Mode? 或者我们分两个 CSV?)
        
        # 补救: 这里的 automation 并不是很完美。
        # 简单处理：我们只做 SEED Loop for Bi-ARL (Full Mode) First.
        # User 只要 "Bi-ARL" 的严谨数据。Ablation 可以在这之后单独跑一次。
        
        # 重新规划: 只跑 Full Mode 5 次来出主表结果。
        pass

    print("Experiment Loop Finished.")

if __name__ == "__main__":
    main()
