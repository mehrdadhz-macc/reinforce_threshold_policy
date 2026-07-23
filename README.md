# REINFORCE Threshold Policy for Intraday Electricity Markets

Replication of **Bertrand & Papavasiliou (2020)** — [*Adaptive Trading in Continuous Intraday Electricity Markets for a Storage Unit*](https://doi.org/10.1109/TPWRS.2019.2957246) (IEEE Transactions on Power Systems, 35(3), 2339–2350). The paper's supplementary appendix (reviewer-response proofs, computational scalability, and learning-stability checks) is kept locally as `appendix_transaction_final_V2.pdf` alongside the main paper PDF -- neither is tracked in this repo, see `.gitignore`.

The paper proposes a parametric threshold policy for a battery storage unit trading on a continuous intraday electricity market (CIM). Policy parameters are learned via REINFORCE. One episode covers a full 24-hour Berlin trading day; the agent trades all delivery hours jointly, tick by tick.

## Project structure

```
project_root/
  data/
    train/
      intraday_auction_curves.csv   # D-1 auction curves (Jan–Jun 2023, seed=42) — generated locally, not tracked
      cim_order_book.csv            # CIM order book — generated locally, not tracked
    test/
      intraday_auction_curves.csv   # D-1 auction curves (Aug 2023, seed=123) — generated locally, not tracked
      cim_order_book.csv            # CIM order book — generated locally, not tracked

  scripts/
    data_generation/
      generate_train_data.py        # synthesise training CIM + auction data
      generate_test_data.py         # synthesise test CIM + auction data
    data_plots/
      plot_auction_curve.py         # visualise D-1 auction MID curve + regimes
    check_data.py                   # validate a dataset

  src/
    data_loader.py       # load_all(), build_day_index(), day_auction_mids()
    environment.py       # MultiHourMarketEnv — episode = one full trading day
    threshold_policy.py  # AlphaPolicy (α₁–α₅, σ_X, σ_Y) + regime computation
    reinforce_trainer.py # REINFORCEAgent — REINFORCE update loop
    parallel_worker.py   # fork-based worker functions for parallel training/eval
    ri_benchmark.py      # rolling-intrinsic LP benchmark (Appendix E)
    training_logger.py   # per-episode metrics + training plots
    eval_plots.py        # six diagnostic evaluation figures

  outputs/
    runs/                # timestamped run directories (model.pt, hparams.txt,
                         #   training/ plots, eval/ plots) — not tracked
    data_plots/          # auction curve visualisations (.gitkeep tracked)

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

**Training data** (seed=42, 2023-01-01 – 2023-06-30, 181 days):
```bash
venv/bin/python3 scripts/data_generation/generate_train_data.py
```

**Test data** (seed=123, 2023-08-01 – 2023-08-31, 31 days, held out after training period):
```bash
venv/bin/python3 scripts/data_generation/generate_test_data.py
```

Both scripts print output file paths, row counts, date range, and spread validation.

**Validate a dataset:**
```bash
venv/bin/python3 scripts/check_data.py train
venv/bin/python3 scripts/check_data.py test
```

**Plot D-1 auction MID curve:**
```bash
# Day 0 of the training split
venv/bin/python3 scripts/data_plots/plot_auction_curve.py --day 0

# Overlay buy/sell regime segmentation
venv/bin/python3 scripts/data_plots/plot_auction_curve.py --day 0 --show-regimes

# Test split, specific day
venv/bin/python3 scripts/data_plots/plot_auction_curve.py --split test --day 5 --show-regimes
```

Figures are saved to `outputs/data_plots/`.

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
| `--out` | `outputs/runs/<timestamp>/model.pt` | Checkpoint path |
| `--seed` | None | RNG seed; fixed defaults used if omitted |
| `--n-levels` | 1 | Quantity discretisation levels per side |
| `--workers` | 1 | Parallel worker processes (Linux/WSL2 only; set to CPU core count for speedup) |

Each run creates a timestamped directory under `outputs/runs/` containing `model.pt`, `hparams.txt`, training plots (`training/`), and evaluation plots (`eval/`). Pass `--out` to override the output path.

## Evaluate

**Quick evaluation** (greedy policy, reports per-day rewards and summary stats):
```bash
venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt
venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt --days 10
```

**Full evaluation** (greedy policy + 6 diagnostic plots, optional RI-LP comparison):
```bash
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt

# Include rolling-intrinsic LP benchmark (slow — one LP solve per tick)
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --with-ri

# Parallel evaluation across CPU cores
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --workers 8
```

When the model path is `outputs/runs/<timestamp>/model.pt`, evaluation plots are saved alongside the model in `outputs/runs/<timestamp>/eval/`. Otherwise they go to `outputs/eval_plots/`.
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

Bertrand, G., & Papavasiliou, A. (2020). Adaptive Trading in Continuous Intraday Electricity Markets for a Storage Unit. *IEEE Transactions on Power Systems*, 35(3), 2339–2350. https://doi.org/10.1109/TPWRS.2019.2957246
