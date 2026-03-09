"""
Statistical Significance Testing for Model Comparison.

This module provides statistical tests to validate that performance improvements
are statistically significant, not just random variance.

Key Tests:
    - Paired t-test: Compare two models across multiple seeds
    - Wilcoxon signed-rank test: Non-parametric alternative
    - Cohen's d: Effect size measurement
    - Confidence intervals: 95% CI for mean differences

Usage Example:
    from src.utils.statistical_tests import compare_models
    
    bi_arl_recalls = [0.711, 0.743, 0.621, 0.775, 0.682]
    rf_recalls = [0.618, 0.618, 0.618, 0.618, 0.618]
    
    result = compare_models(bi_arl_recalls, rf_recalls, metric_name="Recall")
    print(result.summary())
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Results from statistical comparison of two models."""
    
    metric_name: str
    model_a_name: str
    model_b_name: str
    
    # Raw data
    values_a: np.ndarray
    values_b: np.ndarray
    
    # Descriptive statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_diff: float
    
    # Statistical tests
    t_statistic: float
    p_value_ttest: float
    wilcoxon_statistic: Optional[float]
    p_value_wilcoxon: Optional[float]
    
    # Effect size
    cohens_d: float
    
    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if difference is statistically significant."""
        return self.p_value_ttest < alpha
    
    def effect_size_interpretation(self) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(self.cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_symbol = "***" if self.p_value_ttest < 0.001 else \
                     "**" if self.p_value_ttest < 0.01 else \
                     "*" if self.p_value_ttest < 0.05 else "ns"
        
        direction = "higher" if self.mean_diff > 0 else "lower"
        
        summary = f"""
Statistical Comparison: {self.model_a_name} vs {self.model_b_name}
Metric: {self.metric_name}
{'='*60}

Descriptive Statistics:
  {self.model_a_name}: {self.mean_a:.4f} ± {self.std_a:.4f}
  {self.model_b_name}: {self.mean_b:.4f} ± {self.std_b:.4f}
  
  Mean Difference: {abs(self.mean_diff):.4f} ({direction})
  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]

Statistical Tests:
  Paired t-test:
    t = {self.t_statistic:.4f}
    p = {self.p_value_ttest:.4f} {sig_symbol}
    
  Effect Size (Cohen's d): {self.cohens_d:.4f} ({self.effect_size_interpretation()})

Conclusion:
  The difference is {'SIGNIFICANT' if self.is_significant() else 'NOT significant'} at α=0.05.
  {self.model_a_name} is {abs(self.mean_diff):.2%} {direction} than {self.model_b_name}.
  This represents a {self.effect_size_interpretation()} effect.

Significance Levels: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant
{'='*60}
"""
        return summary


def paired_ttest(
    values_a: List[float],
    values_b: List[float]
) -> Tuple[float, float]:
    """
    Perform paired t-test.
    
    Args:
        values_a: Results from model A (N seeds)
        values_b: Results from model B (N seeds, same seeds as A)
        
    Returns:
        (t_statistic, p_value)
    """
    a = np.array(values_a)
    b = np.array(values_b)
    
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length, got {len(a)} vs {len(b)}")
    
    t_stat, p_val = stats.ttest_rel(a, b)
    
    return t_stat, p_val


def wilcoxon_test(
    values_a: List[float],
    values_b: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Perform Wilcoxon signed-rank test (non-parametric).
    
    Args:
        values_a: Results from model A
        values_b: Results from model B
        
    Returns:
        (statistic, p_value) or (None, None) if test cannot be performed
    """
    a = np.array(values_a)
    b = np.array(values_b)
    
    # Check if there are any differences
    diff = a - b
    if np.all(diff == 0):
        return None, None
    
    try:
        stat, p_val = stats.wilcoxon(a, b)
        return stat, p_val
    except:
        return None, None


def cohens_d(
    values_a: List[float],
    values_b: List[float]
) -> float:
    """
    Calculate Cohen's d effect size for paired samples.
    
    Cohen's d interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
        
    Args:
        values_a: Results from model A
        values_b: Results from model B
        
    Returns:
        Cohen's d effect size
    """
    a = np.array(values_a)
    b = np.array(values_b)
    
    # For paired samples, use difference scores
    diff = a - b
    
    # Cohen's d for paired samples = mean_diff / std_diff
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff == 0:
        return 0.0
    
    d = mean_diff / std_diff
    
    return d


def confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.
    
    Args:
        values: Sample values
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound)
    """
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    sem = stats.sem(arr)  # Standard error of mean
    
    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_crit * sem
    
    return (mean - margin, mean + margin)


def compare_models(
    values_a: List[float],
    values_b: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "Performance"
) -> ComparisonResult:
    """
    Comprehensive statistical comparison of two models.
    
    Args:
        values_a: Results from model A (multiple seeds)
        values_b: Results from model B (same seeds)
        model_a_name: Name of model A
        model_b_name: Name of model B
        metric_name: Name of the metric being compared
        
    Returns:
        ComparisonResult object with all statistics
    """
    a = np.array(values_a)
    b = np.array(values_b)
    
    # Descriptive statistics
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a, ddof=1)
    std_b = np.std(b, ddof=1)
    mean_diff = mean_a - mean_b
    
    # Statistical tests
    t_stat, p_ttest = paired_ttest(values_a, values_b)
    w_stat, p_wilcoxon = wilcoxon_test(values_a, values_b)
    
    # Effect size
    d = cohens_d(values_a, values_b)
    
    # Confidence interval of difference
    diff = a - b
    ci_low, ci_high = confidence_interval(diff.tolist())
    
    return ComparisonResult(
        metric_name=metric_name,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        values_a=a,
        values_b=b,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        mean_diff=mean_diff,
        t_statistic=t_stat,
        p_value_ttest=p_ttest,
        wilcoxon_statistic=w_stat,
        p_value_wilcoxon=p_wilcoxon,
        cohens_d=d,
        ci_lower=ci_low,
        ci_upper=ci_high
    )


def compare_all_models(
    results_dict: Dict[str, Dict[str, List[float]]],
    baseline: str = "RandomForest"
) -> Dict[Tuple[str, str], ComparisonResult]:
    """
    Compare all models against a baseline.
    
    Args:
        results_dict: {model_name: {metric_name: [values across seeds]}}
        baseline: Name of baseline model
        
    Returns:
        Dictionary of comparison results: (model, metric) -> ComparisonResult
    """
    comparisons = {}
    
    baseline_results = results_dict.get(baseline)
    if baseline_results is None:
        raise ValueError(f"Baseline '{baseline}' not found in results")
    
    for model_name, model_results in results_dict.items():
        if model_name == baseline:
            continue
        
        for metric_name in model_results.keys():
            if metric_name not in baseline_results:
                continue
            
            comp = compare_models(
                model_results[metric_name],
                baseline_results[metric_name],
                model_a_name=model_name,
                model_b_name=baseline,
                metric_name=metric_name
            )
            
            comparisons[(model_name, metric_name)] = comp
    
    return comparisons


# Example usage
if __name__ == "__main__":
    # Example: Compare Bi-ARL vs Random Forest on Recall
    print("Example: Statistical Comparison\n")
    
    # Simulated results from 5 seeds
    bi_arl_recalls = [0.711, 0.743, 0.621, 0.775, 0.682]
    rf_recalls = [0.618, 0.618, 0.618, 0.618, 0.618]
    
    result = compare_models(
        bi_arl_recalls,
        rf_recalls,
        model_a_name="Bi-ARL",
        model_b_name="Random Forest",
        metric_name="Recall"
    )
    
    print(result.summary())
