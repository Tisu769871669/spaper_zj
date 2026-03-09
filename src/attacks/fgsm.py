"""
Fast Gradient Sign Method (FGSM) Attack

标准对抗攻击实现,用于评估IDS鲁棒性
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FGSMAttack:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015)
    
    生成对抗样本: x_adv = x + epsilon * sign(∇_x L(x, y))
    
    参数:
        model: 目标模型(Defender)
        epsilon: 扰动幅度,默认0.3
        device: 计算设备
    """
    
    def __init__(self, model, epsilon: float = 0.3, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.model.eval()  # 评估模式
    
    def generate(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        生成FGSM对抗样本
        
        参数:
            x: 原始输入 (batch_size, input_dim)
            y_true: 真实标签 (batch_size,)
        
        返回:
            x_adv: 对抗样本
        """
        # 确保输入需要梯度
        x = x.clone().detach().to(self.device)
        x.requires_grad = True
        y_true = y_true.to(self.device)
        
        # Forward pass
        output = self.model(x)
        
        # 计算损失
        loss = F.cross_entropy(output, y_true)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # FGSM: x_adv = x + epsilon * sign(grad)
        data_grad = x.grad.data
        perturbation = self.epsilon * data_grad.sign()
        
        x_adv = x + perturbation
        
        # 裁剪到合法范围[0,1]
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def generate_batch(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 128) -> np.ndarray:
        """
        批量生成对抗样本
        
        参数:
            X: 输入数据 (N, input_dim)
            y: 标签 (N,)
            batch_size: 批大小
        
        返回:
            X_adv: 对抗样本数组
        """
        X_adv_list = []
        
        for i in range(0, len(X), batch_size):
            batch_x = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
            batch_y = torch.LongTensor(y[i:i+batch_size]).to(self.device)
            
            batch_x_adv = self.generate(batch_x, batch_y)
            X_adv_list.append(batch_x_adv.cpu().numpy())
        
        return np.concatenate(X_adv_list, axis=0)


class PGDAttack:
    """
    Projected Gradient Descent Attack (Madry et al., 2018)
    
    迭代版FGSM,更强的攻击
    
    参数:
        model: 目标模型
        epsilon: 总扰动幅度
        alpha: 单步扰动幅度
        steps: 迭代次数
        device: 计算设备
    """
    
    def __init__(self, model, epsilon: float = 0.3, alpha: float = 0.01, 
                 steps: int = 40, device='cuda'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device
        self.model.eval()
    
    def generate(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        生成PGD对抗样本
        
        参数:
            x: 原始输入
            y_true: 真实标签
        
        返回:
            x_adv: 对抗样本
        """
        x = x.clone().detach().to(self.device)
        y_true = y_true.to(self.device)
        
        # 随机初始化
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        
        # PGD迭代
        for _ in range(self.steps):
            x_adv.requires_grad = True
            
            # Forward
            output = self.model(x_adv)
            loss = F.cross_entropy(output, y_true)
            
            # Backward
            self.model.zero_grad()
            loss.backward()
            
            # PGD step
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                
                # Project回epsilon球内
                perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + perturbation, 0, 1).detach()
        
        return x_adv
    
    def generate_batch(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 128) -> np.ndarray:
        """批量生成PGD对抗样本"""
        X_adv_list = []
        
        for i in range(0, len(X), batch_size):
            batch_x = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
            batch_y = torch.LongTensor(y[i:i+batch_size]).to(self.device)
            
            batch_x_adv = self.generate(batch_x, batch_y)
            X_adv_list.append(batch_x_adv.cpu().numpy())
        
        return np.concatenate(X_adv_list, axis=0)


# 便捷函数
def apply_fgsm(model, X, y, epsilon=0.3, device='cuda'):
    """
    便捷函数:对数据应用FGSM攻击
    
    使用示例:
        X_adv = apply_fgsm(defender_model, X_test, y_test, epsilon=0.3)
    """
    attack = FGSMAttack(model, epsilon=epsilon, device=device)
    return attack.generate_batch(X, y)


def apply_pgd(model, X, y, epsilon=0.3, alpha=0.01, steps=40, device='cuda'):
    """
    便捷函数:对数据应用PGD攻击
    """
    attack = PGDAttack(model, epsilon=epsilon, alpha=alpha, 
                      steps=steps, device=device)
    return attack.generate_batch(X, y)
