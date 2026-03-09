# Scripts for Bi-ARL Experiments

Scripts directory contains automated experiment runners and utilities.

## Quick Start

### 1. Run Bi-level Training
```bash
python scripts/train_bilevel.sh --seed 42 --episodes 300
```

### 2. Compare Bi-level vs Multi-Agent RL
```bash
python scripts/compare_bilevel_vs_marl.py
```

This will:
- Train both paradigms with seeds [42, 101, 202]
- Save TensorBoard logs to `runs/`
- Generate comparison CSV

### 3. View Training Progress
```bash
tensorboard --logdir runs/
```

## Available Scripts

- `compare_bilevel_vs_marl.py` - Compare Bi-level vs standard Multi-Agent RL
- (More scripts will be added as optimization progresses)

## Expected Output

After running comparison script, you should see:
- Inner Loop convergence messages (Bi-level only)
- Separate reward curves for Inner/Outer loops
- Final model checkpoints in `src/checkpoints/`
