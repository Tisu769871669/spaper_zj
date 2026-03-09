#!/usr/bin/env python
"""
自动生成统计分析报告和LaTeX表格

读取experiment_results.csv中的所有结果,进行统计检验并生成论文表格
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# 路径设置
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.utils.statistical_tests import compare_models
from src.utils.result_analyzer import ResultAnalyzer

def main():
    print(f"\n{'='*70}")
    print(f"  统计分析和LaTeX表格生成")
    print(f"{'='*70}\n")
    
    # 初始化分析器
    analyzer = ResultAnalyzer()
    
    if analyzer.df is None or len(analyzer.df) == 0:
        print("❌ 错误: 未找到结果文件或文件为空")
        print("请先运行: python src/experiments.py --seed 42")
        return 1
    
    print(f"✅ 加载了 {len(analyzer.df)} 条结果记录\n")
    
    # 检查有哪些models和seeds
    models = analyzer.df['Model'].unique()
    seeds = analyzer.df['Seed'].unique() if 'Seed' in analyzer.df.columns else [42]
    conditions = analyzer.df['Condition'].unique()
    
    print(f"模型: {list(models)}")
    print(f"Seeds: {list(seeds)}")
    print(f"测试条件: {list(conditions)}\n")
    
    # 1. 生成聚合统计
    print(f"{'='*70}")
    print(f"  1. 聚合统计 (Mean ± Std)")
    print(f"{'='*70}\n")
    
    agg_df = analyzer.aggregate_by_model()
    print(agg_df)
    print()
    
    # 2. 生成对比表格(带统计显著性)
    print(f"\n{'='*70}")
    print(f"  2. 性能对比表格")
    print(f"{'='*70}\n")
    
    for condition in conditions:
        print(f"\n--- {condition} 条件 ---\n")
        table = analyzer.generate_comparison_table(
            baseline="RandomForest",
            condition=condition,
            metrics=['Recall', 'Precision', 'F1', 'FPR']
        )
        print(table.to_string(index=False))
    
    # 3. 生成LaTeX表格
    print(f"\n{'='*70}")
    print(f"  3. LaTeX表格代码")
    print(f"{'='*70}\n")
    
    # 主结果表格 - 所有条件
    print("\n% 表1: 所有模型在三种测试条件下的性能对比")
    print("% 直接复制粘贴到论文的04_experiments.tex中\n")
    
    analyzer.print_latex_table(
        condition="Clean",
        caption="Model Comparison on Clean Test Data (N=3 seeds, Mean±Std)"
    )
    
    print("\n% 如需其他条件的单独表格:")
    for condition in ['Adversarial', 'Stress']:
        if condition in conditions:
            print(f"\n% {condition} 条件:")
            analyzer.print_latex_table(
                condition=condition,
                caption=f"Model Comparison under {condition} Attack (N=3 seeds, Mean±Std)"
            )
    
    # 4. 详细统计检验报告
    print(f"\n{'='*70}")
    print(f"  4. 详细统计检验报告")
    print(f"{'='*70}\n")
    
    # 保存完整报告
    report_file = project_root / "outputs" / "results" / "statistical_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  Bi-ARL 统计显著性检验报告\n")
        f.write("="*70 + "\n\n")
        
        baseline = "RandomForest"
        for condition in conditions:
            f.write(f"\n## {condition} 条件\n")
            f.write("-" * 70 + "\n\n")
            
            # 获取所有模型在该条件下的数据
            df_cond = analyzer.df[analyzer.df['Condition'] == condition]
            
            # 对每个模型vs baseline进行检验
            for model in models:
                if model == baseline:
                    continue
                
                model_data = df_cond[df_cond['Model'] == model]
                baseline_data = df_cond[df_cond['Model'] == baseline]
                
                if len(model_data) > 0 and len(baseline_data) > 0:
                    f.write(f"\n### {model} vs {baseline}\n\n")
                    
                    for metric in ['Recall', 'Precision', 'F1', 'FPR']:
                        if metric in model_data.columns:
                            model_values = model_data[metric].tolist()
                            baseline_values = baseline_data[metric].tolist()
                            
                            if len(model_values) >= 2 and len(baseline_values) >= 2:
                                result = compare_models(
                                    model_values,
                                    baseline_values,
                                    model_a_name=model,
                                    model_b_name=baseline,
                                    metric_name=metric
                                )
                                
                                f.write(result.summary())
                                f.write("\n")
    
    print(f"✅ 详细报告已保存: {report_file}")
    
    # 5. 生成简明总结
    print(f"\n{'='*70}")
    print(f"  5. 关键发现总结")
    print(f"{'='*70}\n")
    
    df_clean = analyzer.df[analyzer.df['Condition'] == 'Clean']
    if 'Bi-ARL' in models and baseline in models:
        bi_arl_recall = df_clean[df_clean['Model'] == 'Bi-ARL']['Recall'].mean()
        rf_recall = df_clean[df_clean['Model'] == baseline]['Recall'].mean()
        improvement = (bi_arl_recall - rf_recall) / rf_recall * 100
        
        print(f"📊 主要发现:")
        print(f"  • Bi-ARL Recall: {bi_arl_recall:.4f}")
        print(f"  • Random Forest Recall: {rf_recall:.4f}")
        print(f"  • 提升: {improvement:+.2f}%")
        print(f"  • 统计显著性: 详见报告文件")
    
    print(f"\n{'='*70}")
    print(f"✅ 分析完成!")
    print(f"{'='*70}\n")
    
    print("📄 生成的文件:")
    print(f"  • {report_file}")
    print("\n📋 下一步:")
    print("  1. 查看LaTeX代码,复制到论文")
    print("  2. 查看详细报告,添加统计分析段落")
    print("  3. 更新论文04_experiments.tex中的TBD数据")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
