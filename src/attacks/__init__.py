"""
Adversarial Attacks Module

提供标准对抗攻击方法用于评估IDS鲁棒性
"""

from .fgsm import FGSMAttack, PGDAttack, apply_fgsm, apply_pgd

__all__ = [
    'FGSMAttack',
    'PGDAttack', 
    'apply_fgsm',
    'apply_pgd'
]
