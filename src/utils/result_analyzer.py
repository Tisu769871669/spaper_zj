"""
Result Analysis and Visualization Tools.

This module provides tools to analyze experimental results and generate
publication-quality tables and figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Path setup
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.statistical_tests import compare_models, compare_all_models


class ResultAnalyzer:
    """
    Analyzes experimental results and generates comparison tables.
    """
    
    def __init__(self, results_csv: str = "src/experiment_results.csv"):
        """
        Initialize analyzer.
        
        Args:
            results_csv: Path to results CSV file
        """
        self.results_path = Path(project_root) / results_csv
        self.df = None
        
        if self.results_path.exists():
            self.df = pd.read_csv(self.results_path)
            print(f"Loaded {len(self.df)} result rows from {results_csv}")
        else:
            print(f"Warning: Results file not found: {results_csv}")
    
    def aggregate_by_model(self) -> pd.DataFrame:
        """
        Aggregate results by model (mean ± std across seeds).
        
        Returns:
            DataFrame with aggregated statistics
        """
        if self.df is None:
            return pd.DataFrame()
        
        # Group by Model and Condition
        grouped = self.df.groupby(['Model', 'Condition'])
        
        # Calculate mean and std
        agg_df = grouped.agg({
            'Acc': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'Precision': ['mean', 'std'],
            'F1': ['mean', 'std'],
            'FPR': ['mean', 'std']
        })
        
        return agg_df
    
    def generate_comparison_table(
        self,
        baseline: str = "RandomForest",
        condition: str = "Clean",
        metrics: List[str] = ['Acc', 'Recall', 'F1', 'FPR']
    ) -> pd.DataFrame:
        """
        Generate comparison table with statistical significance.
        
        Args:
            baseline: Baseline model name
            condition: Test condition (Clean, Adv, Stress)
            metrics: Metrics to include
            
        Returns:
            Comparison table DataFrame
        """
        if self.df is None:
            return pd.DataFrame()
        
        # Filter by condition
        df_cond = self.df[self.df['Condition'] == condition]
        
        #Build results dict
        results_dict = {}
        for model in df_cond['Model'].unique():
            model_df = df_cond[df_cond['Model'] == model]
            results_dict[model] = {
                metric: model_df[metric].tolist()
                for metric in metrics
            }
        
        # Generate table rows
        rows = []
        for model in results_dict.keys():
            row = {'Model': model}
            
            for metric in metrics:
                values = results_dict[model][metric]
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                
                # Compare with baseline if not baseline
                if model != baseline and baseline in results_dict:
                    comp = compare_models(
                        values,
                        results_dict[baseline][metric],
                        model_a_name=model,
                        model_b_name=baseline,
                        metric_name=metric
                    )
                    
                    # Add significance stars
                    if comp.p_value_ttest < 0.001:
                        sig = "***"
                    elif comp.p_value_ttest < 0.01:
                        sig = "**"
                    elif comp.p_value_ttest < 0.05:
                        sig = "*"
                    else:
                        sig = ""
                    
                    row[metric] = f"{mean:.4f}±{std:.4f}{sig}"
                else:
                    row[metric] = f"{mean:.4f}±{std:.4f}"
            
            rows.append(row)
        
        table_df = pd.DataFrame(rows)
        return table_df
    
    def print_latex_table(
        self,
        condition: str = "Clean",
        caption: str = "Model Comparison on Clean Data"
    ):
        """
        Generate LaTeX table code.
        
        Args:
            condition: Test condition
            caption: Table caption
        """
        table_df = self.generate_comparison_table(condition=condition)
        
        if table_df.empty:
            print("No data available for LaTeX table")
            return
        
        print(f"\n% LaTeX Table: {caption}")
        print("\\begin{table}[h]")
        print("\\centering")
        print(f"\\caption{{{caption}}}")
        print("\\label{tab:results_" + condition.lower() + "}")
        print("\\begin{tabular}{l" + "c" * (len(table_df.columns) - 1) + "}")
        print("\\hline")
        
        # Header
        print(" & ".join(table_df.columns) + " \\\\")
        print("\\hline")
        
        # Rows
        for _, row in table_df.iterrows():
            print(" & ".join(map(str, row.values)) + " \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
        print(f"\n% Significance: * p<0.05, ** p<0.01, *** p<0.001\n")


def main():
    """Main analysis function."""
    print("="*60)
    print("  Experimental Results Analysis")
    print("="*60)
    
    analyzer = ResultAnalyzer()
    
    if analyzer.df is None:
        print("\n⚠️  No results file found. Run experiments first:")
        print("   python src/experiments.py --seed 42")
        return
    
    # Print summary statistics
    print("\n1. Summary Statistics:")
    print(analyzer.aggregate_by_model())
    
    # Generate comparison tables
    print("\n2. Comparison Table (Clean Data):")
    table = analyzer.generate_comparison_table(condition="Clean")
    print(table)
    
    # Generate LaTeX output
    print("\n3. LaTeX Table:")
    analyzer.print_latex_table(
        condition="Clean",
        caption="Model Comparison on Clean Test Data (N=3 seeds, Mean±Std)"
    )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
