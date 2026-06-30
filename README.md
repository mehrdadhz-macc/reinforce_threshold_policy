# REINFORCE Threshold Policy for Intraday Electricity Markets

Replication of **Bertrand & Papavasiliou (2020)** — *Adaptive Trading in Continuous Intraday Electricity Markets for a Storage Unit*.

The paper proposes a parametric threshold policy for a battery storage unit trading on a continuous intraday electricity market (CIM). Policy parameters are learned via REINFORCE. One episode covers a full 24-hour Berlin trading day; the agent trades all delivery hours jointly, tick by tick.

## Project structure

```
project_root/
  data/
    train/
      intraday_auction_curves.csv   # D-1 auction curves (Jan 2023, seed=42)
      cim_order_book.csv            # CIM order book — not tracked (>800 MB)
    test/
      intraday_auction_curves.csv   # D-1 auction curves (Apr 2023, seed=123)
      cim_order_book.csv            # CIM order book — not tracked (>3 GB)

  scripts/
    data_generation/
      generate_train_data.py        # synthesise training CIM + auction data
      generate_test_data.py         # synthesise test CIM + auction data
    check_data.py                   # validate a dataset

  src/
    data_loader.py       # load_all(), build_day_index(), day_auction_mids()
    environment.py       # MultiHourMarketEnv — episode = one full trading day
    threshold_policy.py  # AlphaPolicy (α₁–α₅, σ_X, σ_Y) + regime computation
    reinforce_trainer.py # REINFORCEAgent — REINFORCE update loop
    ri_benchmark.py      # rolling-intrinsic LP benchmark (Appendix E)
    training_logger.py   # per-episode metrics + training plots
    eval_plots.py        # six diagnostic evaluation figures

  outputs/
    models/              # saved policy checkpoints (.pt)
    plots/               # training figures
    eval_plots/          # evaluation figures
    logs/                # training logs

  train.py               # training entry point
  test.py                # quick greedy evaluation (rewards only)
  evaluate.py            # full evaluation with diagnostic plots
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # numpy, pandas, torch, scipy, matplotlib
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

Both scripts print output file paths, row counts, date range, and spread validation.

**Validate a dataset:**
```bash
venv/bin/python3 scripts/check_data.py train
venv/bin/python3 scripts/check_data.py test
```

## Train

```bash
# Default: 200 days × 4 reps, finest tick resolution
venv/bin/python3 train.py

# Full paper curriculum (hourly → 15-min → 5-min, §VI-A)
venv/bin/python3 train.py --curriculum --days 200 --reps 4

# Custom output path and learning rate
venv/bin/python3 train.py --days 200 --lr 5e-4 --out outputs/models/my_policy.pt

# Random hyperparameter seed (samples α₁–α₅, σ from uniform ranges)
venv/bin/python3 train.py --seed 7
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--days` | 200 | Training days (paper §VI-A uses 200) |
| `--reps` | 4 | Repetitions over training days per phase |
| `--curriculum` | off | Three-phase frequency curriculum |
| `--lr` | 1e-3 | Adam learning rate |
| `--out` | `outputs/models/policy_multihour.pt` | Checkpoint path |
| `--seed` | None | RNG seed; fixed defaults used if omitted |
| `--n-levels` | 1 | Quantity discretisation levels per side |

Saves a `.pt` checkpoint containing `state_dict`, `param_snapshot`, and training metadata.

## Evaluate

**Quick evaluation** (greedy policy, reports per-day rewards and summary stats):
```bash
venv/bin/python3 test.py --model outputs/models/policy_multihour.pt
venv/bin/python3 test.py --model outputs/models/policy_multihour.pt --days 10
```

**Full evaluation** (greedy policy + 6 diagnostic plots, optional RI-LP comparison):
```bash
venv/bin/python3 evaluate.py --model outputs/models/policy_multihour.pt

# Include rolling-intrinsic LP benchmark (slow — one LP solve per tick)
venv/bin/python3 evaluate.py --model outputs/models/policy_multihour.pt --with-ri
```

Evaluation plots saved to `outputs/eval_plots/`:
1. Performance distribution — histogram of policy vs RI-LP rewards
2. Daily reward time series
3. Intraday trading behaviour — price landscape + position buildup for example days
4. SoC trajectory heatmap
5. Threshold vs market price scatter (µ_Y vs bid, µ_X vs ask)
6. Revenue by delivery hour (hours 0–23)

## Policy overview

The `AlphaPolicy` computes buy (µ_X) and sell (µ_Y) threshold means for each active delivery hour at each tick via four steps (paper §V-B – §V-E):

1. **Auction anchor** — centres thresholds within the D-1 auction price range
2. **SoC pressure** — adjusts thresholds based on implied end-of-day storage level
3. **Urgency** — exponentially increases aggressiveness as delivery approaches
4. **RI reference** — reallocates probability mass using the rolling-intrinsic LP signal

Actions are sampled from a Gaussian threshold distribution (paper §IV-B) and updated by undiscounted REINFORCE (paper §IV-A, Eq. 2).

## Reference

Bertrand, G., & Papavasiliou, A. (2020). Adaptive Trading in Continuous Intraday Electricity Markets for a Storage Unit. *IEEE Transactions on Power Systems*, 35(3), 2339–2350.
