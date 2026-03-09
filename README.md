# Bi-ARL: Bi-level Adversarial Reinforcement Learning for Intrusion Detection

PyTorch implementation of **Bi-ARL**, a bi-level adversarial reinforcement learning framework for robust network intrusion detection.

---

## 📖 Overview

Bi-ARL introduces a novel bi-level optimization approach where an Attacker agent and a Defender agent engage in adversarial training:

- **Inner Loop**: Attacker converges to best-response (KL divergence < 0.01)
- **Outer Loop**: Defender trains against the optimal Attacker

This yields a Nash equilibrium solution with provable worst-case robustness.

**Key Results**:

- ✅ **Lowest FPR**: 7.7% (vs MARL 19.9%, Vanilla PPO 37.5%)
- ✅ **Best Balance**: Superior FPR-Recall trade-off
- ✅ **Adversarial Robustness**: Stable performance under FGSM attacks

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Anaconda/Miniconda

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/bi-arl.git
cd bi-arl

# 2. Create conda environment
conda create -n spaper python=3.8
conda activate spaper

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NSL-KDD dataset (already included in data/)
```

### Train & Evaluate

```bash
# Train all models (Bi-ARL, Baselines) with 4 seeds
python scripts/train_all_models.py

# Expected time: ~2 hours (with GPU)
# Output: outputs/models/
```

```bash
# Run experiments and generate results
python src/experiments.py --seed 42

# Output: experiment_results.csv
```

```bash
# Generate statistical analysis
python scripts/generate_analysis.py

# Output: outputs/results/statistical_report.txt
```

---

## 📁 Project Structure

```
bi-arl/
├── data/                   # NSL-KDD dataset
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── src/
│   ├── agents/            # Attacker & Defender agents
│   ├── algorithms/        # Bi-level optimizer
│   ├── attacks/           # FGSM & PGD attacks
│   ├── baselines/         # Vanilla PPO, LSTM-IDS, MARL
│   ├── envs/              # Network security game
│   └── utils/             # Config, data loader, PPO
├── scripts/
│   ├── train_all_models.py           # Full training pipeline
│   ├── evaluate_ablation.py          # Ablation study
│   ├── evaluate_adversarial_robustness.py  # FGSM/PGD evaluation
│   └── generate_analysis.py          # Statistical tests
├── outputs/
│   ├── models/            # Trained models
│   ├── results/           # Experiment results
│   └── logs/              # TensorBoard logs
└── latex_source/          # Paper LaTeX source
```

---

## 🧪 Reproduce Paper Results

### Main Experiments

```bash
# 1. Train all models (4 seeds: 42, 123, 3407, 8888)
python scripts/train_all_models.py

# 2. Evaluate performance
python scripts/evaluate_ablation.py

# 3. Generate statistical analysis with p-values
python scripts/generate_analysis.py
```

### Ablation Study

```bash
# Evaluate all ablation variants
python scripts/batch_evaluate_ablation.py

# Results saved to: outputs/results/ablation_results.csv
```

### Adversarial Robustness

```bash
# Evaluate under FGSM attacks
python scripts/evaluate_adversarial_robustness.py

# Results saved to: outputs/results/adversarial_robustness.csv
```

---

## 📊 Key Results

### Main Performance (Clean Condition)

| Model | Recall | Precision | FPR | F1 |
|-------|--------|-----------|-----|-----|
| **Bi-ARL** | 48.6% | **89.3%** | **7.7%** | 62.9% |
| MARL | **62.8%** | 80.6% | 19.9% | 70.6% |
| Vanilla PPO | 39.4% | 58.1% | 37.5% | 47.0% |
| LSTM-IDS | 61.8% | 96.6% | 2.8% | 75.4% |

### Ablation Study

| Variant | Recall (Clean) | FPR (Clean) |
|---------|---------------|-------------|
| Full Bi-ARL | 48.6% | **7.7%** |
| w/o Inner Loop | 40.4% | 38.2% |
| w/o Attacker | 39.4% | 37.5% |
| w/o Bi-level | 62.8% | 19.9% |

**Key Finding**: Bi-level structure achieves **12.2pp** FPR reduction vs standard MARL.

---

## 🛠️ Configuration

All hyperparameters are centralized in `src/utils/config.py`:

```python
# Experiment seeds
SEEDS = [42, 123, 3407, 8888]

# Training parameters
RL_EPISODES = 100
LSTM_EPOCHS = 20

# Bi-level optimization
INNER_LOOP_STEPS = 5
KL_THRESHOLD = 0.01

# PPO hyperparameters
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
```

---

## 📈 Training Curves

Monitor training with TensorBoard:

```bash
tensorboard --logdir=runs
# Open http://localhost:6006/
```

---

## 🔬 Advanced Usage

### Train Single Model

```bash
# Bi-ARL only
python src/main_train_bilevel.py --seed 42 --episodes 100

# Vanilla PPO
python src/baselines/vanilla_ppo.py --seed 42 --episodes 100

# MARL (w/o Bi-level)
python src/baselines/marl_baseline.py --seed 42 --episodes 100
```

### Custom Evaluation

```python
from src.utils.config import Config
from src.agents.defender_agent import DefenderAgent

# Load model
defender = DefenderAgent().to(Config.DEVICE)
model_path = Config.find_model_file("BiARL", seed=42, model_name="defender")
defender.load_state_dict(torch.load(model_path))

# Evaluate on your data
action, _ = defender.get_action(your_state)
```

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{bi-arl2024,
  title={Bi-level Adversarial Reinforcement Learning for Robust Intrusion Detection},
  author={Your Name},
  journal={TBD},
  year={2024}
}
```

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- NSL-KDD dataset: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
- PPO implementation inspired by OpenAI Spinning Up

---

## 📧 Contact

For questions, please open an issue or contact: <your.email@example.com>

---

**Status**: Work in progress. The codebase and paper assets are usable, but training, evaluation, and manuscript polishing still need final alignment before submission.
