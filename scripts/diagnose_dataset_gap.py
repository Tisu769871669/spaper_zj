"""
诊断为什么RL在不同数据集上表现差异巨大

关键指标：
1. 数据集难度 - 树模型性能 (如果树99%说明数据本身特征区分度太高)
2. 是否存在"好的对抗空间" - 有多少样本能被成功对抗
3. RL策略的学习曲线 (是否收敛，收敛到哪里)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 读取当前结果
results_dir = Path("outputs/results")

print("="*80)
print("数据集难度与RL失败原因诊断")
print("="*80)

# 1. 查看树模型vs RL的性能差距
for dataset in ["nsl_kdd", "unsw_nb15", "cic_ids2017_random"]:
    csv_path = results_dir / f"ablation_results_detailed_{dataset}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        print(f"\n【{dataset.upper()}】")
        print("-" * 60)
        
        # 聚合结果
        summary = df.groupby("model")[["f1", "precision", "recall", "fpr"]].agg(["mean", "std"])
        print(summary)
        
        # 计算难度指标
        tree_best = df[df["model"].str.contains("HGBT|XGBoost|LightGBM", case=False)]["f1"].max()
        rl_best = df[df["model"].str.contains("BiARL|PPO", case=False)]["f1"].max()
        
        gap = tree_best - rl_best
        print(f"\n📊 关键指标:")
        print(f"  - 树模型最佳F1: {tree_best:.2%}")
        print(f"  - RL最佳F1: {rl_best:.2%}")
        print(f"  - 性能差距: {gap:.2%}")
        
        if tree_best > 0.95:
            print(f"  ⚠️  数据集特征可区分度非常高 (树模型已接近饱和)")
            print(f"      → RL难以通过对抗学习获得额外收益")
        elif gap > 0.3:
            print(f"  ⚠️  RL策略存在根本性问题:")
            print(f"      → 检查reward设计是否合理")
            print(f"      → 检查action space是否太小")
            print(f"      → 检查超参 (noise_std, inner_loop_steps)")

print("\n" + "="*80)
print("建议:")
print("="*80)
print("""
如果NSL-KDD上有效但其他数据集上无效，问题很可能是：

❌ 原因1: 数据集特征区分度不同
   NSL-KDD是较老的数据集，特征区分模式单一
   现代数据集特征混杂，树模型能捕捉复杂非线性→ RL无法超越

✅ 解决: 考虑在噪声更大的场景测试 (e.g., 混合攻击、对抗防御)

❌ 原因2: RL action space/reward太弱
   如果attacker action都是小范围扰动，可能无法充分探索
   
✅ 解决: 增强action diversity或改进reward signal

❌ 原因3: 超参数未对现代数据集优化
   NSL-KDD用的inner_loop_steps=5, noise_std=0.1
   其他数据集用的noise_std=0.03/0.02 (太小)
   
✅ 解决: 做一个小规模超参数搜索 (只用UNSW子集快速验证)
""")
