#!/usr/bin/env python
"""
FPR优化后的自动评估和对比脚本

训练完成后运行此脚本,自动对比新旧模型的性能
"""

import sys
from pathlib import Path

print("\n" + "="*70)
print("  FPR优化效果评估")
print("="*70 + "\n")

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🔄 步骤1: 运行所有seeds评估...")
print("-" * 70)

import subprocess

seeds = [42, 101, 202]
for seed in seeds:
    print(f"\n评估 Seed {seed}...")
    cmd = [
        sys.executable,
        str(project_root / 'src' / 'experiments.py'),
        '--seed', str(seed)
    ]
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode == 0:
        print(f"✅ Seed {seed} 评估完成")
    else:
        print(f"❌ Seed {seed} 评估失败")

print("\n" + "="*70)
print("🔄 步骤2: 生成统计分析...")
print("="*70 + "\n")

cmd = [sys.executable, str(project_root / 'scripts' / 'generate_analysis.py')]
subprocess.run(cmd, cwd=project_root)

print("\n" + "="*70)
print("📊 步骤3: FPR优化前后对比")
print("="*70 + "\n")

# 读取结果
import pandas as pd

results_file = project_root / 'src' / 'experiment_results.csv'
if results_file.exists():
    df = pd.read_csv(results_file)
    
    # 筛选Bi-ARL结果
    bi_arl = df[df['Model'] == 'Bi-ARL']
    
    if len(bi_arl) > 0:
        print("Bi-ARL 性能指标 (Mean±Std):\n")
        
        for condition in bi_arl['Condition'].unique():
            data = bi_arl[bi_arl['Condition'] == condition]
            print(f"{condition} 条件:")
            print(f"  FPR:       {data['FPR'].mean():.4f} ± {data['FPR'].std():.4f}")
            print(f"  Recall:    {data['Recall'].mean():.4f} ± {data['Recall'].std():.4f}")
            print(f"  Precision: {data['Precision'].mean():.4f} ± {data['Precision'].std():.4f}")
            print(f"  F1:        {data['F1'].mean():.4f} ± {data['F1'].std():.4f}")
            print()
        
        # 检查目标是否达成
        print("="*70)
        print("🎯 目标达成情况:")
        print("="*70)
        
        clean_fpr = bi_arl[bi_arl['Condition'] == 'Clean']['FPR'].mean()
        stress_fpr = bi_arl[bi_arl['Condition'] == 'Stress']['FPR'].mean()
        clean_recall = bi_arl[bi_arl['Condition'] == 'Clean']['Recall'].mean()
        stress_recall = bi_arl[bi_arl['Condition'] == 'Stress']['Recall'].mean()
        
        fpr_clean_ok = "✅" if clean_fpr < 0.15 else "❌"
        fpr_stress_ok = "✅" if stress_fpr < 0.20 else "❌"
        recall_clean_ok = "✅" if clean_recall > 0.60 else "❌"
        recall_stress_ok = "✅" if stress_recall > 0.60 else "❌"
        
        print(f"1. FPR (Clean) < 15%:    {clean_fpr*100:.1f}% {fpr_clean_ok}")
        print(f"2. FPR (Stress) < 20%:   {stress_fpr*100:.1f}% {fpr_stress_ok}")
        print(f"3. Recall (Clean) > 60%: {clean_recall*100:.1f}% {recall_clean_ok}")
        print(f"4. Recall (Stress) > 60%: {stress_recall*100:.1f}% {recall_stress_ok}")
        
        all_ok = all([
            clean_fpr < 0.15,
            stress_fpr < 0.20,
            clean_recall > 0.60,
            stress_recall > 0.60
        ])
        
        print("\n" + "="*70)
        if all_ok:
            print("🎉 优化成功!所有目标均已达成!")
        else:
            print("⚠️  部分目标未达成,可能需要进一步调整")
            print("\n建议:")
            if clean_fpr >= 0.15 or stress_fpr >= 0.20:
                print("  • FPR仍偏高,考虑增加FP惩罚或降低FN惩罚")
            if clean_recall <= 0.60 or stress_recall <= 0.60:
                print("  • Recall偏低,考虑略微增加FN惩罚")
        print("="*70)
    else:
        print("❌ 未找到Bi-ARL的评估结果")
else:
    print("❌ 结果文件不存在,请先运行评估")

print("\n✅ 评估完成!")
print(f"详细报告: {project_root / 'outputs' / 'results' / 'statistical_report.txt'}\n")
