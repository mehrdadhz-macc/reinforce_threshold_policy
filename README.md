# B&P REINFORCE Threshold Policy

Replication of Bertrand & Papavasiliou (2020) — *Adaptive Trading in Continuous
Intraday Electricity Markets for a Storage Unit*.

## Project structure

```
project_root/
  data/
    train/          # generated training data (seed=42,  Jan 2023)
    test/           # generated test data    (seed=123, Apr 2023)

  scripts/
    data_generation/
      generate_train_data.py
      generate_test_data.py
    check_data.py   # validate a dataset

  src/
    data_loader.py      # load_all(split="train"|"test")
    storage.py          # StorageUnit + Action
    environment.py      # MarketEnv (one episode = one delivery hour)
    threshold_policy.py # AlphaPolicy (α₁–α₅, σ_X, σ_Y)
    reinforce_trainer.py# REINFORCEAgent (REINFORCE update loop)

  outputs/
    models/         # saved policy checkpoints (.pt)
    plots/          # figures
    logs/           # training logs

  train.py          # main training entry point
```

## Generate data

**Training data** (seed=42, 2023-01-01 – 2023-01-30):
```bash
venv/bin/python3 scripts/data_generation/generate_train_data.py
```

**Test data** (seed=123, 2023-04-01 – 2023-04-30):
```bash
venv/bin/python3 scripts/data_generation/generate_test_data.py
```

Both scripts print: output file paths, row counts, date range, and spread
validation result. They do not overwrite each other's output.

## Load datasets

```python
from src.data_loader import load_all

cim_train, auc_train = load_all(split="train")
cim_test,  auc_test  = load_all(split="test")

# Or with explicit paths:
cim, auc = load_all(cim_path="my/cim.csv", auction_path="my/auc.csv")
```

## Validate a dataset

```bash
venv/bin/python3 scripts/check_data.py train
venv/bin/python3 scripts/check_data.py test
```

## Run training

```bash
# 50 episodes (default)
venv/bin/python3 train.py

# Full training set, custom learning rate, custom output path
venv/bin/python3 train.py --episodes 720 --lr 5e-4 --out outputs/models/policy_full.pt
```

Prints one line per episode (`reward`, `loss`, `n_steps`). Saves a `.pt` file
containing `state_dict`, `param_snapshot`, and training metadata.
