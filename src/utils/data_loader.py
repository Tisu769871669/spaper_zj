
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.utils.config import Config


class BaseTabularLoader:
    def __init__(self, train_path=None, test_path=None):
        self.train_path = str(train_path or Config.DATASET_PATH)
        self.test_path = str(test_path or Config.TEST_DATASET_PATH)
        self.scaler = MinMaxScaler()
        self.encoders = {}
        self.cat_cols = []
        self.label_col = "label"
        self.drop_cols = []
        self.feature_dim = Config.STATE_DIM

    def has_real_data(self):
        return os.path.exists(self.train_path) and os.path.exists(self.test_path)

    def load_data(self, mode='train'):
        """
        加载数据集。
        Args:
            mode (str): 'train' (KDDTrain+) or 'test' (KDDTest+)
        Returns:
            X (np.array): 特征矩阵 (归一化后)
            y (np.array): 标签 (二分类: 0=正常, 1=攻击)
        """
        path = self.train_path if mode == 'train' else self.test_path
        
        if os.path.exists(path):
            print(f"正在从 {path} 加载 {Config.DATASET_NAME} ({mode}) 数据...")
            try:
                df = self._read_split(path)

                # 测试集依赖训练集的编码器和归一化统计量。
                # 如果当前实例还没fit过,先自动加载训练集完成fit。
                if mode == 'test' and not self._is_fitted():
                    train_df = self._read_split(self.train_path)
                    self.preprocess(train_df, fit=True)

                if mode == 'train':
                    return self.preprocess(df, fit=True)
                else:
                    return self.preprocess(df, fit=False)
            except Exception as e:
                print(f"读取文件出错: {e}. 回退到模拟数据。")
                return self.generate_synthetic_data()
        else:
            print(f"未在 {path} 找到数据集。使用模拟数据进行验证。")
            return self.generate_synthetic_data()

    def _read_split(self, path):
        raise NotImplementedError

    def _is_fitted(self):
        return hasattr(self.scaler, "scale_") and all(col in self.encoders for col in self.cat_cols)

    def preprocess(self, df, fit=True):
        df = df.copy()

        # 1. 编码类别特征 (Label Encoding)
        for col in self.cat_cols:
            if col in df.columns:
                if fit:
                    # 训练阶段: 学习映射关系
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    # 测试阶段: 使用已有的映射
                    if col in self.encoders:
                        le = self.encoders[col]
                        # 处理未知标签: 将未见过的类别映射为 -1 或其他已知类别
                        # 这里简单处理: 如果遇到新标签，替换为该列的众数(mode)或 0
                        # 更好的方式是使用 OrdinalEncoder(handle_unknown='use_encoded_value')
                        # 这里手动实现一个安全的 transform
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[col] = le.transform(df[col])
                    else:
                        # 如果没有对应的 encoder (不应发生)，则重新 fit (不推荐)
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
        
        # 2. 分离标签
        y = self.build_binary_labels(df)
        
        # 3. 从 X 中删除标签及辅助列
        drop_cols = [col for col in [self.label_col, *self.drop_cols] if col in df.columns]
        X = df.drop(drop_cols, axis=1)
        
        # 4. 归一化 (MinMax Scaling)
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        self.feature_dim = X.shape[1]
        return X.astype(np.float32), y

    def build_binary_labels(self, df):
        return df[self.label_col].apply(lambda x: 0 if str(x).startswith('normal') else 1).values

    def generate_synthetic_data(self, n_samples=1024):
        """在真实数据不可用时生成最小可运行的占位数据。"""
        rng = np.random.default_rng(Config.DEFAULT_SEED)
        X = rng.random((n_samples, Config.STATE_DIM), dtype=np.float32)
        y = rng.integers(0, 2, size=n_samples, endpoint=False)
        return X, y


class NSLKDDLoader(BaseTabularLoader):
    def __init__(self, train_path=None, test_path=None):
        super().__init__(train_path=train_path, test_path=test_path)
        self.cat_cols = ['protocol_type', 'service', 'flag']
        self.columns = [
            "duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
        ]

    def _read_split(self, path):
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :42].copy()
        df.columns = self.columns
        return df


class UNSWNB15Loader(BaseTabularLoader):
    def __init__(self, train_path=None, test_path=None):
        super().__init__(train_path=train_path, test_path=test_path)
        self.cat_cols = ['proto', 'service', 'state']
        self.label_col = 'label'
        self.drop_cols = ['id', 'attack_cat']

    def _read_split(self, path):
        df = pd.read_csv(path)
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def build_binary_labels(self, df):
        if self.label_col in df.columns:
            return df[self.label_col].astype(int).values
        raise ValueError(f"UNSW-NB15 split missing label column: {self.label_col}")


class CICIDS2017Loader(BaseTabularLoader):
    TRAIN_FILES = [
        "Monday-WorkingHours.pcap_ISCX.csv.parquet",
        "Tuesday-WorkingHours.pcap_ISCX.csv.parquet",
        "Wednesday-workingHours.pcap_ISCX.csv.parquet",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet",
    ]
    TEST_FILES = [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
    ]

    def __init__(self, root_dir=None):
        root = root_dir or Config.DATASET_PATH
        super().__init__(train_path=root, test_path=root)
        self.root_dir = str(root)
        self.label_col = "Label"
        self.drop_cols = []
        self._cached_random_split = None

    def has_real_data(self):
        expected = self.TRAIN_FILES + self.TEST_FILES
        return all(os.path.exists(os.path.join(self.root_dir, name)) for name in expected)

    def _resolve_files(self, mode):
        if Config.SPLIT_MODE == "random_stratified":
            filenames = self.TRAIN_FILES + self.TEST_FILES
        else:
            filenames = self.TRAIN_FILES if mode == "train" else self.TEST_FILES
        return [os.path.join(self.root_dir, name) for name in filenames]

    def _load_combined_df(self):
        files = self._resolve_files("train")
        frames = [pd.read_parquet(path) for path in files]
        df = pd.concat(frames, ignore_index=True)
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _get_random_split(self):
        if self._cached_random_split is not None:
            return self._cached_random_split

        df = self._load_combined_df()
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = [col for col in df.columns if col != self.label_col]
        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median) else median)

        labels = self.build_binary_labels(df)
        train_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SIZE,
            random_state=Config.DEFAULT_SEED,
            stratify=labels,
        )
        self._cached_random_split = (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
        return self._cached_random_split

    def _read_split(self, path):
        raise NotImplementedError("CIC-IDS2017 uses multi-file split loading.")

    def load_data(self, mode='train'):
        files = self._resolve_files(mode)
        missing = [path for path in files if not os.path.exists(path)]
        if missing:
            print(f"CIC-IDS2017 缺失文件: {missing[:3]}")
            return self.generate_synthetic_data()

        print(f"正在从 {self.root_dir} 加载 {Config.DATASET_NAME} ({mode}) 数据...")
        if Config.SPLIT_MODE == "random_stratified":
            train_df, test_df = self._get_random_split()
            df = train_df if mode == "train" else test_df
            if mode == "test" and not self._is_fitted():
                self.preprocess(train_df, fit=True)
        else:
            frames = [pd.read_parquet(path) for path in files]
            df = pd.concat(frames, ignore_index=True)
            df.columns = [str(col).strip() for col in df.columns]

            if mode == "test" and not self._is_fitted():
                train_df = pd.concat([pd.read_parquet(path) for path in self._resolve_files("train")], ignore_index=True)
                train_df.columns = [str(col).strip() for col in train_df.columns]
                self.preprocess(train_df, fit=True)

        return self.preprocess(df, fit=(mode == "train"))

    def preprocess(self, df, fit=True):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = [col for col in df.columns if col != self.label_col]
        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median) else median)

        max_samples = Config.MAX_TRAIN_SAMPLES if fit else Config.MAX_TEST_SAMPLES
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=Config.DEFAULT_SEED).reset_index(drop=True)

        return super().preprocess(df, fit=fit)

    def build_binary_labels(self, df):
        labels = df[self.label_col].astype(str).str.strip().str.upper()
        return (labels != "BENIGN").astype(int).values


class CICIoT2023Loader(BaseTabularLoader):
    def __init__(self, root_dir=None):
        root = root_dir or Config.DATASET_PATH
        super().__init__(train_path=root, test_path=root)
        self.root_dir = str(root)
        self.label_col = "__label__"
        self.drop_cols = []
        self._cached_random_split = None
        self._csv_files = None

    def _discover_csv_files(self):
        if self._csv_files is not None:
            return self._csv_files
        if not os.path.exists(self.root_dir):
            self._csv_files = []
            return self._csv_files
        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for name in filenames:
                if name.lower().endswith(".csv"):
                    files.append(os.path.join(root, name))
        self._csv_files = sorted(files)
        return self._csv_files

    def has_real_data(self):
        return len(self._discover_csv_files()) > 0

    def _is_benign_file(self, path):
        return "benign" in os.path.basename(path).lower()

    def _file_family(self, path):
        stem = Path(path).stem
        return re.sub(r"\d+\.pcap$", ".pcap", stem)

    def _read_labeled_csv(self, path, nrows=None):
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        df.columns = [str(col).strip() for col in df.columns]
        df[self.label_col] = 0 if self._is_benign_file(path) else 1
        return df

    def _load_balanced_sample(self):
        files = self._discover_csv_files()
        benign_files = [path for path in files if self._is_benign_file(path)]
        attack_files = [path for path in files if not self._is_benign_file(path)]
        if not benign_files or not attack_files:
            raise FileNotFoundError(
                "CICIoT2023 requires at least one benign CSV and one attack CSV under data/CICIoT2023."
            )

        total_target = (Config.MAX_TRAIN_SAMPLES or 0) + (Config.MAX_TEST_SAMPLES or 0)
        total_target = total_target or 360000
        per_class_target = total_target // 2

        benign_df = self._load_class_subset(benign_files, per_class_target)
        attack_df = self._load_class_subset(attack_files, per_class_target)
        df = pd.concat([benign_df, attack_df], ignore_index=True)
        return df.sample(frac=1.0, random_state=Config.DEFAULT_SEED).reset_index(drop=True)

    def _load_class_subset(self, files, target_rows):
        rows_per_file = max(1, int(np.ceil(target_rows / max(len(files), 1))))
        frames = [self._read_labeled_csv(path, nrows=rows_per_file) for path in files]
        df = pd.concat(frames, ignore_index=True)
        if len(df) > target_rows:
            df = df.sample(n=target_rows, random_state=Config.DEFAULT_SEED).reset_index(drop=True)
        return df

    def _get_random_split(self):
        if self._cached_random_split is not None:
            return self._cached_random_split

        df = self._load_balanced_sample()
        labels = self.build_binary_labels(df)
        train_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SIZE,
            random_state=Config.DEFAULT_SEED,
            stratify=labels,
        )
        self._cached_random_split = (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
        return self._cached_random_split

    def _get_grouped_file_split(self):
        if self._cached_random_split is not None:
            return self._cached_random_split

        files = self._discover_csv_files()
        families = {}
        for path in files:
            families.setdefault(self._file_family(path), []).append(path)

        train_files = []
        test_files = []
        for family, family_files in sorted(families.items()):
            family_files = sorted(family_files)
            if len(family_files) == 1:
                if self._is_benign_file(family_files[0]):
                    train_files.append(family_files[0])
                else:
                    test_files.append(family_files[0])
                continue
            split_idx = max(1, int(np.ceil(len(family_files) * (1 - Config.TEST_SIZE))))
            split_idx = min(split_idx, len(family_files) - 1)
            train_files.extend(family_files[:split_idx])
            test_files.extend(family_files[split_idx:])

        if not train_files or not test_files:
            raise RuntimeError("Grouped CICIoT2023 split failed to create both train and test file sets.")

        train_df = self._load_class_subset(train_files, Config.MAX_TRAIN_SAMPLES or 240000)
        test_df = self._load_class_subset(test_files, Config.MAX_TEST_SAMPLES or 120000)
        self._cached_random_split = (
            train_df.sample(frac=1.0, random_state=Config.DEFAULT_SEED).reset_index(drop=True),
            test_df.sample(frac=1.0, random_state=Config.DEFAULT_SEED).reset_index(drop=True),
        )
        return self._cached_random_split

    def _read_split(self, path):
        raise NotImplementedError("CICIoT2023 uses directory-based split loading.")

    def load_data(self, mode='train'):
        if not self.has_real_data():
            print(f"CICIoT2023 未找到 CSV 文件: {self.root_dir}")
            return self.generate_synthetic_data()

        print(f"正在从 {self.root_dir} 加载 {Config.DATASET_NAME} ({mode}) 数据...")
        if Config.SPLIT_MODE == "grouped_file_holdout":
            train_df, test_df = self._get_grouped_file_split()
        else:
            train_df, test_df = self._get_random_split()
        df = train_df if mode == "train" else test_df
        if mode == "test" and not self._is_fitted():
            self.preprocess(train_df, fit=True)
        return self.preprocess(df, fit=(mode == "train"))

    def preprocess(self, df, fit=True):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        for col in df.columns:
            if col == self.label_col:
                continue
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(median) else median)

        return super().preprocess(df, fit=fit)

    def build_binary_labels(self, df):
        return df[self.label_col].astype(int).values


class CSECICIDS2018Loader(BaseTabularLoader):
    """
    Loader for CSE-CIC-IDS2018 dataset.

    Expected directory layout (data/CSE_CIC_IDS2018/):
        02-14-2018.csv
        02-15-2018.csv
        ...  (any *.csv files in the directory)

    Column format matches CIC-IDS2017: numeric flow features + 'Label' column.
    Benign rows are labelled 'Benign'; attack rows have various attack-type strings.

    Data source: https://www.unb.ca/cic/datasets/ids-2018.html
                 AWS S3 bucket s3://cse-cic-ids2018/
    """

    def __init__(self, root_dir=None):
        root = root_dir or Config.DATASET_PATH
        super().__init__(train_path=root, test_path=root)
        self.root_dir = str(root)
        self.label_col = "Label"
        self.drop_cols = []
        self._cached_split = None

    def _discover_csv_files(self):
        if not os.path.exists(self.root_dir):
            return []
        files = sorted(
            str(p)
            for p in Path(self.root_dir).glob("*.csv")
        )
        return files

    def has_real_data(self):
        return len(self._discover_csv_files()) > 0

    def _load_all(self) -> pd.DataFrame:
        files = self._discover_csv_files()
        frames = []
        for path in files:
            try:
                df = pd.read_csv(path, low_memory=False)
                df.columns = [str(c).strip() for c in df.columns]
                frames.append(df)
            except Exception as exc:
                print(f"  [CIC-IDS2018] Skipping {path}: {exc}")
        if not frames:
            raise FileNotFoundError(f"No readable CSV files in {self.root_dir}")
        return pd.concat(frames, ignore_index=True)

    def _load_sampled(self, max_total: int) -> pd.DataFrame:
        files = self._discover_csv_files()
        if not files:
            raise FileNotFoundError(f"No readable CSV files in {self.root_dir}")

        per_file_target = max(1, int(np.ceil(max_total / len(files))))
        frames = []
        for path in files:
            try:
                df = pd.read_csv(path, low_memory=False)
                df.columns = [str(c).strip() for c in df.columns]
                if len(df) > per_file_target:
                    labels = self.build_binary_labels(df)
                    df, _ = train_test_split(
                        df,
                        train_size=per_file_target,
                        random_state=Config.DEFAULT_SEED,
                        stratify=labels,
                    )
                frames.append(df.reset_index(drop=True))
            except Exception as exc:
                print(f"  [CIC-IDS2018] Skipping {path}: {exc}")

        if not frames:
            raise FileNotFoundError(f"No readable CSV files in {self.root_dir}")

        df = pd.concat(frames, ignore_index=True)
        if len(df) > max_total:
            labels = self.build_binary_labels(df)
            df, _ = train_test_split(
                df,
                train_size=max_total,
                random_state=Config.DEFAULT_SEED,
                stratify=labels,
            )
            df = df.reset_index(drop=True)
        return df

    def _get_split(self):
        if self._cached_split is not None:
            return self._cached_split

        max_total = (Config.MAX_TRAIN_SAMPLES or 300000) + (Config.MAX_TEST_SAMPLES or 150000)
        df = self._load_sampled(max_total)
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = [c for c in df.columns if c != self.label_col]
        # CIC-IDS2018 CSVs contain mixed string/numeric columns in some files.
        # Coerce all feature columns to numeric before imputation and splitting.
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in numeric_cols:
            if df[col].isna().any():
                med = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(med) else med)

        labels = self.build_binary_labels(df)
        train_df, test_df = train_test_split(
            df,
            test_size=Config.TEST_SIZE,
            random_state=Config.DEFAULT_SEED,
            stratify=labels,
        )
        self._cached_split = (
            train_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )
        return self._cached_split

    def _read_split(self, path):
        raise NotImplementedError("CSECICIDS2018 uses directory-based loading.")

    def load_data(self, mode="train"):
        if not self.has_real_data():
            print(f"[CIC-IDS2018] No CSV files found in: {self.root_dir}")
            return self.generate_synthetic_data()

        print(f"Loading CSE-CIC-IDS2018 ({mode}) from {self.root_dir} ...")
        train_df, test_df = self._get_split()
        df = train_df if mode == "train" else test_df
        if mode == "test" and not self._is_fitted():
            self.preprocess(train_df, fit=True)
        return self.preprocess(df, fit=(mode == "train"))

    def preprocess(self, df, fit=True):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if col == self.label_col:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                med = df[col].median()
                df[col] = df[col].fillna(0 if pd.isna(med) else med)
        return super().preprocess(df, fit=fit)

    def build_binary_labels(self, df):
        labels = df[self.label_col].astype(str).str.strip().str.upper()
        return (labels != "BENIGN").astype(int).values


def build_data_loader(dataset_name=None):
    dataset_key = (dataset_name or Config.DATASET_NAME).lower()
    if dataset_key == "nsl-kdd":
        return NSLKDDLoader()
    if dataset_key == "unsw-nb15":
        return UNSWNB15Loader()
    if dataset_key == "cic-ids2017":
        return CICIDS2017Loader()
    if dataset_key == "cic-ids2017-random":
        return CICIDS2017Loader()
    if dataset_key == "ciciot2023":
        return CICIoT2023Loader()
    if dataset_key == "ciciot2023-grouped":
        return CICIoT2023Loader()
    if dataset_key == "cse-cic-ids2018":
        return CSECICIDS2018Loader()
    raise ValueError(f"Unsupported dataset loader: {dataset_key}")
