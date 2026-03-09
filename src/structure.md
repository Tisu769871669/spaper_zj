# 代码结构说明

本项目实现了用于网络威胁检测的 **双层对抗强化学习 (Bi-level Adversarial Reinforcement Learning)** 框架。

## 目录布局

- **`envs/`**: 自定义 OpenAI Gym/Gymnasium 环境。
  - `network_security_game.py`: 核心环境，模拟攻防博弈场景。
- **`agents/`**: 强化学习智能体。
  - `attacker_agent.py`: 控制红队（攻击者）的智能体。
  - `defender_agent.py`: 控制蓝队（防御者）的智能体。包含内层“语义模块”的占位符。
- **`utils/`**: 辅助工具。
  - `config.py`: 超参数和配置常量。
- **`main_train.py`**: 训练循环的主程序入口。

## 核心类

- `NetworkSecurityGame`:
    - 观测空间 (Observation Space): 网络状态 (流量特征, 主机日志)。
    - 动作空间 (Action Space):
        - 攻击者: [Exploit (漏洞利用), Obfuscate (混淆), Idle (静默), ...]
        - 防御者: [Block IP (封锁IP), Isolate Host (隔离主机), Analyze Semantic (语义分析), ...]
- `BiLevelDefender` (计划中):
    - 实现内层循环逻辑 (LLM 约束)。
