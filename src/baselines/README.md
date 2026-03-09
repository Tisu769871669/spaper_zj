# Baselines for Bi-ARL

This directory contains baseline implementations for comparison experiments.

## Available Baselines

### 1. Vanilla PPO Defender (`vanilla_ppo.py`)
**Purpose**: Demonstrate the value of adversarial training

**Description**: 
- Single-agent PPO Defender trained on **clean data only**
- No attacker present during training
- Represents standard RL approach without adversarial robustification

**Usage**:
```bash
python src/baselines/vanilla_ppo.py --seed 42 --episodes 300
```

**Expected Result**: 
Good performance on clean data, but lower robustness under adversarial attacks compared to Bi-ARL.

---

### 2. LSTM-IDS (`lstm_ids.py`)
**Purpose**: Represent traditional deep learning IDS approaches

**Description**:
- Supervised learning with LSTM architecture
- 2-layer LSTM with 64 hidden units
- Trained on labeled NSL-KDD data

**Usage**:
```bash
python src/baselines/lstm_ids.py --seed 42 --epochs 30
```

**Expected Result**:
Competitive accuracy on clean data, but may lack adversarial robustness.

---

## Comparison Matrix

| Model | Training Paradigm | Advers. Training? | Clean Acc | Adv. Recall | Purpose |
|-------|------------------|-------------------|-----------|-------------|---------|
| **Bi-ARL (Ours)** | Bi-level RL | ✅ Yes | ~75% | **~71%** | Robust IDS |
| Vanilla PPO | Single-agent RL | ❌ No | ~72% | ~60% | Show adv. value |
| LSTM-IDS | Supervised DL | ❌ No | ~76% | ~68% | DL baseline |
| Random Forest | Traditional ML | ❌ No | ~77% | ~62% | Simple baseline |

---

## Adding New Baselines

To add a new baseline:

1. Create `src/baselines/your_baseline.py`
2. Implement training script with `--seed` argument
3. Save model to `src/your_baseline_seed{seed}.pth`
4. Update this README
5. Add to `scripts/run_all_baselines.sh`

---

## Notes

- All baselines should use the **same random seeds** for fair comparison
- Save models with naming convention: `{model_name}_seed{seed}.pth`
- Use `src/utils/data_loader.py` for consistent data loading
