import torch
import numpy as np
import random
from pathlib import Path

class Config:
    """
    项目全局配置
    
    包含所有超参数、路径、训练设置等
    """
    
    # ==================== 项目路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # 数据路径
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_DATA = DATA_DIR / "KDDTrain+.txt"
    TEST_DATA = DATA_DIR / "KDDTest+.txt"
    UNSW_TRAIN_DATA = DATA_DIR / "UNSW_NB15_training-set.csv"
    UNSW_TEST_DATA = DATA_DIR / "UNSW_NB15_testing-set.csv"
    CIC_IDS2017_DIR = DATA_DIR / "CIC_IDS2017_machine_learning"
    CICIOT2023_DIR = DATA_DIR / "CICIoT2023"
    CSE_CIC_IDS2018_DIR = DATA_DIR / "CSE_CIC_IDS2018"
    
    # 输出路径
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = OUTPUT_DIR / "models"
    LOGS_DIR = OUTPUT_DIR / "logs"
    RESULTS_DIR = OUTPUT_DIR / "results"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
    FIGURES_DIR = RESULTS_DIR / "figures"
    
    # 临时/兼容路径(用于向后兼容)
    SRC_DIR = PROJECT_ROOT / "src"
    
    # ==================== 实验配置 ====================
    DATASET_PROFILES = {
        "nsl-kdd": {
            "train_path": TRAIN_DATA,
            "test_path": TEST_DATA,
            "state_dim": 41,
            "inner_loop_steps": 5,
            "attacker_noise_std": 0.1,
            "categorical_feature_indices": [1, 2, 3],
            "reward_profile": {
                "tp_def": 1.0,
                "tn_def": 0.5,
                "fp_def": -1.0,
                "fn_def": -2.0,
                "tp_att": -1.0,
                "tn_att": 0.0,
                "fp_att": 1.0,
                "fn_att": 2.0,
            },
        },
        "unsw-nb15": {
            "train_path": UNSW_TRAIN_DATA,
            "test_path": UNSW_TEST_DATA,
            "state_dim": 42,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.03,
            "categorical_feature_indices": [1, 2, 3],
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
        "cic-ids2017": {
            "train_path": CIC_IDS2017_DIR,
            "test_path": CIC_IDS2017_DIR,
            "state_dim": 78,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.02,
            "categorical_feature_indices": [],
            "max_train_samples": 300000,
            "max_test_samples": 150000,
            "split_mode": "day_split",
            "test_size": 0.33,
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
        "cic-ids2017-random": {
            "train_path": CIC_IDS2017_DIR,
            "test_path": CIC_IDS2017_DIR,
            "state_dim": 78,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.02,
            "categorical_feature_indices": [],
            "max_train_samples": 300000,
            "max_test_samples": 150000,
            "split_mode": "random_stratified",
            "test_size": 0.2,
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
        "ciciot2023": {
            "train_path": CICIOT2023_DIR,
            "test_path": CICIOT2023_DIR,
            "state_dim": 39,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.01,
            "categorical_feature_indices": [],
            "max_train_samples": 240000,
            "max_test_samples": 120000,
            "split_mode": "random_stratified",
            "test_size": 0.2,
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
        "ciciot2023-grouped": {
            "train_path": CICIOT2023_DIR,
            "test_path": CICIOT2023_DIR,
            "state_dim": 39,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.01,
            "categorical_feature_indices": [],
            "max_train_samples": 240000,
            "max_test_samples": 120000,
            "split_mode": "grouped_file_holdout",
            "test_size": 0.2,
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
        "cse-cic-ids2018": {
            "train_path": CSE_CIC_IDS2018_DIR,
            "test_path": CSE_CIC_IDS2018_DIR,
            "state_dim": 78,
            "inner_loop_steps": 1,
            "attacker_noise_std": 0.02,
            "categorical_feature_indices": [],
            "max_train_samples": 300000,
            "max_test_samples": 150000,
            "split_mode": "random_stratified",
            "test_size": 0.2,
            "reward_profile": {
                "tp_def": 1.5,
                "tn_def": 1.0,
                "fp_def": -2.5,
                "fn_def": -2.0,
                "tp_att": -1.5,
                "tn_att": 0.0,
                "fp_att": 2.5,
                "fn_att": 2.0,
            },
        },
    }
    DATASET_NAME = "nsl-kdd"

    # 随机种子列表(用于多次实验)
    SEEDS = [42, 3407, 8888, 123, 2026]
    DEFAULT_SEED = 42
    
    # 训练参数
    RL_EPISODES = 100  # RL训练轮数
    LSTM_EPOCHS = 20   # LSTM训练周期
    
    # ==================== 模型路径辅助方法 ====================
    @classmethod
    def dataset_slug(cls) -> str:
        return cls.DATASET_NAME.replace("-", "_")

    @classmethod
    def get_model_path(cls, model_type: str, seed: int, model_name: str = "model") -> Path:
        """
        获取模型保存路径
        
        Args:
            model_type: 模型类型 ("BiARL", "VanillaPPO", "LSTM", etc.)
            seed: 随机种子
            model_name: 模型名称 ("attacker", "defender", "model")
            
        Returns:
            模型文件路径
        """
        model_dir = cls.MODELS_DIR / cls.dataset_slug() / model_type / f"seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{model_name}.pth"
    
    @classmethod
    def get_checkpoint_path(cls, model_type: str, seed: int, episode: int) -> Path:
        """获取检查点路径"""
        ckpt_dir = cls.CHECKPOINTS_DIR / cls.dataset_slug() / model_type / f"seed{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / f"checkpoint_ep{episode}.pth"
    
    @classmethod
    def get_log_path(cls, model_type: str, seed: int) -> Path:
        """获取日志路径"""
        log_dir = cls.LOGS_DIR / cls.dataset_slug() / model_type
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"seed{seed}.json"
    
    @classmethod
    def get_results_csv(cls) -> Path:
        """获取结果CSV路径"""
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.RESULTS_DIR / f"experiment_results_{cls.dataset_slug()}.csv"
    
    # ==================== 环境配置 ====================
    # 状态和动作空间维度
    STATE_DIM = DATASET_PROFILES[DATASET_NAME]["state_dim"]
    DEFENDER_ACTION_DIM = 10
    ATTACKER_ACTION_DIM = 10
    
    # 原始维度(向后兼容)
    ACTION_DIM_DEFENDER = 10
    ACTION_DIM_ATTACKER = 10
    
    # 环境参数
    MAX_STEPS = 50
    BALANCED_ENV_SAMPLING = True
    
    # Dataset paths (向后兼容)
    DATASET_PATH = str(DATASET_PROFILES[DATASET_NAME]["train_path"])
    TEST_DATASET_PATH = str(DATASET_PROFILES[DATASET_NAME]["test_path"])
    REWARD_PROFILE = DATASET_PROFILES[DATASET_NAME]["reward_profile"]
    
    # ==================== PPO 超参数 ====================
    LR = 3e-4
    GAMMA = 0.99
    EPS_CLIP = 0.2
    K_EPOCHS = 4  # PPO更新epoch数
    BATCH_SIZE = 64
    
    # ==================== Bi-level 优化参数 ====================
    INNER_LOOP_STEPS = 5      # 内层循环最大步数
    KL_THRESHOLD = 0.01       # KL散度收敛阈值
    USE_BILEVEL = True        # 是否使用双层优化
    ATTACKER_NOISE_STD = DATASET_PROFILES[DATASET_NAME]["attacker_noise_std"]
    CATEGORICAL_FEATURE_INDICES = DATASET_PROFILES[DATASET_NAME]["categorical_feature_indices"]
    MAX_TRAIN_SAMPLES = DATASET_PROFILES[DATASET_NAME].get("max_train_samples")
    MAX_TEST_SAMPLES = DATASET_PROFILES[DATASET_NAME].get("max_test_samples")
    SPLIT_MODE = DATASET_PROFILES[DATASET_NAME].get("split_mode", "official")
    TEST_SIZE = DATASET_PROFILES[DATASET_NAME].get("test_size", 0.2)
    
    # ==================== 内层循环稳定化参数 ====================
    # 攻击者熵正则化系数（鼓励探索，防止策略坍缩）
    ENTROPY_COEFF_INIT = 0.05       # 初始熵系数
    ENTROPY_COEFF_MIN = 0.001       # 最小熵系数
    ENTROPY_DECAY = 0.995           # 每 episode 的退火乘数
    # 攻击者热身调度（训练初期限制内层步数）
    WARMUP_RATIO = 0.2              # 前 20% 的 episode 使用 K_inner=1
    WARMUP_KINNER = 1               # 热身期间的内层步数
    
    # ==================== 设备配置 ====================
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Configuring Setup on Device: {DEVICE}")
    
    # Random seed (向后兼容)
    RANDOM_SEED = 42
    
    # ==================== 随机种子设置 ====================
    @staticmethod
    def set_seed(seed: int = 42):
        """
        设置全局随机种子
        
        Args:
            seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保可复现性(可能略微影响性能)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Global Random Seed set to: {seed}")

    @classmethod
    def configure_dataset(cls, dataset_name: str):
        dataset_key = dataset_name.lower()
        if dataset_key not in cls.DATASET_PROFILES:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(cls.DATASET_PROFILES)}")

        profile = cls.DATASET_PROFILES[dataset_key]
        cls.DATASET_NAME = dataset_key
        cls.STATE_DIM = profile["state_dim"]
        cls.DATASET_PATH = str(profile["train_path"])
        cls.TEST_DATASET_PATH = str(profile["test_path"])
        cls.REWARD_PROFILE = profile["reward_profile"]
        cls.INNER_LOOP_STEPS = profile.get("inner_loop_steps", cls.INNER_LOOP_STEPS)
        cls.ATTACKER_NOISE_STD = profile.get("attacker_noise_std", cls.ATTACKER_NOISE_STD)
        cls.CATEGORICAL_FEATURE_INDICES = profile.get(
            "categorical_feature_indices",
            cls.CATEGORICAL_FEATURE_INDICES,
        )
        cls.MAX_TRAIN_SAMPLES = profile.get("max_train_samples")
        cls.MAX_TEST_SAMPLES = profile.get("max_test_samples")
        cls.SPLIT_MODE = profile.get("split_mode", cls.SPLIT_MODE)
        cls.TEST_SIZE = profile.get("test_size", cls.TEST_SIZE)
        print(f"Dataset configured: {cls.DATASET_NAME} (state_dim={cls.STATE_DIM})")
    
    # ==================== 模型加载辅助(向后兼容) ====================
    @classmethod
    def find_model_file(cls, model_type: str, seed: int, model_name: str = "defender") -> Path:
        """
        智能查找模型文件(支持新旧路径)
        
        优先级:
        1. outputs/models/{model_type}/seed{seed}/{model_name}.pth
        2. src/{model_name}_{model_type}_seed{seed}.pth (旧格式)
        3. src/{model_name}.pth (legacy)
        
        Returns:
            找到的模型路径,如果都不存在则返回优先路径
        """
        # 新格式路径
        new_path = cls.get_model_path(model_type, seed, model_name)
        if new_path.exists():
            return new_path

        # 兼容旧版无数据集隔离路径
        legacy_datasetless_path = cls.MODELS_DIR / model_type / f"seed{seed}" / f"{model_name}.pth"
        if cls.DATASET_NAME == "nsl-kdd" and legacy_datasetless_path.exists():
            return legacy_datasetless_path
        
        # 旧格式路径
        if model_type == "BiARL":
            old_path = cls.SRC_DIR / f"{model_name}_bilevel_seed{seed}.pth"
        elif model_type == "VanillaPPO":
            old_path = cls.SRC_DIR / f"defender_vanilla_ppo_seed{seed}.pth"
        elif model_type == "LSTM":
            old_path = cls.SRC_DIR / f"lstm_ids_seed{seed}.pth"
        else:
            old_path = cls.SRC_DIR / f"{model_name}_{model_type}_seed{seed}.pth"
        
        if old_path.exists():
            return old_path
        
        # Legacy路径
        legacy_path = cls.SRC_DIR / f"{model_name}.pth"
        if legacy_path.exists():
            return legacy_path
        
        # 都不存在,返回新格式路径(调用者可以用于保存)
        return new_path
