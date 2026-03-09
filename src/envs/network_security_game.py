
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.utils.config import Config
from src.utils.data_loader import build_data_loader

class NetworkSecurityGame(gym.Env):
    """
    自定义网络安全对抗环境。
    模拟攻击者和防御者互动的双层优化问题。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(NetworkSecurityGame, self).__init__()
        
        self.state_dim = Config.STATE_DIM
        self.action_dim_attacker = Config.ACTION_DIM_ATTACKER
        self.action_dim_defender = Config.ACTION_DIM_DEFENDER
        
        # 加载数据
        self.loader = build_data_loader(Config.DATASET_NAME)
        self.dataset_X, self.dataset_y = self.loader.load_data()
        self.data_index = 0
        self.class_indices = {
            0: np.where(self.dataset_y == 0)[0],
            1: np.where(self.dataset_y == 1)[0],
        }
        
        # 定义动作空间
        # 为了简单起见，我们暂时假设离散动作
        self.attacker_action_space = spaces.Discrete(self.action_dim_attacker)
        self.defender_action_space = spaces.Discrete(self.action_dim_defender)
        
        # 观测空间 (网络状态)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.state = None

    def _sample_index(self):
        if Config.BALANCED_ENV_SAMPLING:
            target_label = np.random.randint(0, 2)
            candidate_indices = self.class_indices.get(target_label)
            if candidate_indices is not None and len(candidate_indices) > 0:
                return int(np.random.choice(candidate_indices))
        return int(np.random.randint(0, len(self.dataset_X)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 从数据集中随机采样一个状态
        idx = self._sample_index()
        self.state = self.dataset_X[idx]
        self.current_label = self.dataset_y[idx] # 记录是攻击还是正常流量
        
        return self.state, {}

    def get_defender_observation(self, attacker_action, state=None, label=None):
        """
        根据攻击者动作生成防御者观测到的扰动状态。
        """
        base_state = self.state if state is None else state
        current_label = self.current_label if label is None else label
        modified_state = np.asarray(base_state, dtype=np.float32).copy()

        if current_label == 1:
            start_idx = (attacker_action % 10) * 4
            end_idx = min(start_idx + 4, self.state_dim)
            target_indices = np.arange(start_idx, end_idx)
            valid_indices = [
                idx for idx in target_indices
                if idx not in Config.CATEGORICAL_FEATURE_INDICES
            ]
            if valid_indices:
                noise = np.random.normal(0, Config.ATTACKER_NOISE_STD, size=len(valid_indices))
                modified_state[valid_indices] += noise
            modified_state = np.clip(modified_state, 0, 1)

        return modified_state.astype(np.float32)

    def step(self, action_dict):
        """
        Step 函数: 执行攻防博弈逻辑。
        1. 攻击者: 根据动作对当前流量特征添加扰动 (Evasion Attack)。
        2. 防御者: 根据(被扰动后的)特征做出决策 (放行/封锁/深度检查)。
        3. 结算奖励。
        """
        attacker_action = action_dict.get('attacker')
        defender_action = action_dict.get('defender')
        
        self.current_step += 1
        
        modified_state = action_dict.get('modified_state')
        if modified_state is None:
            modified_state = self.get_defender_observation(attacker_action)
            
        # --- 2. 防御者行动 (Defender Step) ---
        # Defender 观测到的是 modified_state
        # Action 定义:
        # 0-3: 放行 (Predict Normal) - 信心度不同
        # 4-7: 封锁 (Predict Attack) - 信心度不同
        # 8-9: 深度语义检查 (Inner Loop) -> 稍后由主循环特定逻辑触发或在此增加消耗
        
        # 简化映射:
        if defender_action < 5:
            prediction = 0 # 判定为正常
        else:
            prediction = 1 # 判定为攻击
            
        # --- 3. 奖励计算 (Reward Calculation) ---
        # 奖励矩阵:
        # TP (攻击, 判攻击): +1 (Def), -1 (Att)
        # TN (正常, 判正常): +0.1 (Def), 0 (Att) -> 鼓励正确但不显着
        # FP (正常, 判攻击): -1 (Def/误报), +1 (Att/诱骗成功)
        # FN (攻击, 判正常): -5 (Def/漏报-严重), +5 (Att/逃逸成功-高奖赏)
        
        r_def = 0
        r_att = 0
        
        true_label = self.current_label
        
        reward_profile = Config.REWARD_PROFILE
        if true_label == 1 and prediction == 1: # TP
            r_def = reward_profile["tp_def"]
            r_att = reward_profile["tp_att"]
        elif true_label == 0 and prediction == 0: # TN
            r_def = reward_profile["tn_def"]
            r_att = reward_profile["tn_att"]
        elif true_label == 0 and prediction == 1: # FP (误报)
            r_def = reward_profile["fp_def"]
            r_att = reward_profile["fp_att"]
        elif true_label == 1 and prediction == 0: # FN (漏报)
            r_def = reward_profile["fn_def"]
            r_att = reward_profile["fn_att"]
            
        # --- 状态更新 ---
        # 采样下一个数据
        idx = self._sample_index()
        self.state = self.dataset_X[idx]
        self.current_label = self.dataset_y[idx]
        
        terminated = self.current_step >= Config.MAX_STEPS
        truncated = False
        
        info = {
            'attacker_action': attacker_action,
            'defender_action': defender_action,
            'true_label': true_label,
            'prediction': prediction,
            'is_success': (true_label == prediction)
        }
        
        rewards = {
            'attacker': r_att,
            'defender': r_def
        }
        
        return self.state, rewards, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, State: {self.state[:3]}...")
