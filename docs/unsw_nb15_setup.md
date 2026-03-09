# UNSW-NB15 Setup

## Official source

- Dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset

The official UNSW page states that the partitioned files are:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Expected local paths

Place the two files under `data/`:

- `data/UNSW_NB15_training-set.csv`
- `data/UNSW_NB15_testing-set.csv`

## Supported commands

Train Bi-ARL on UNSW-NB15:

```bash
python src/main_train_bilevel.py --dataset unsw-nb15 --seed 42 --episodes 100
```

Train vanilla PPO:

```bash
python src/baselines/vanilla_ppo.py --dataset unsw-nb15 --seed 42 --episodes 100
```

Train LSTM baseline:

```bash
python src/baselines/lstm_ids.py --dataset unsw-nb15 --seed 42 --epochs 20
```

Run the new boosting baseline:

```bash
python src/baselines/hgbt_ids.py --dataset unsw-nb15 --seed 42
```

Evaluate main results:

```bash
python scripts/evaluate_main_results.py --dataset unsw-nb15
```

## Notes

- The current RL environment assumes binary detection and discrete attacker/defender actions, so UNSW-NB15 is mapped to binary labels via the official `label` column.
- The loader drops `id` and `attack_cat` and uses the remaining tabular features.
- Current feature dimension for this pipeline is `42`.
