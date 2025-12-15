# PowerPlay: Optimal Energy Storage Decisions in ERCOT

PowerPlay is a profit maximization framework for Independent Power Producers (IPP) operating Battery Energy Storage Systems (BESS) in the ERCOT market. This repository contains the implementation of adaptive control frameworks designed to optimize a 4-hour Lithium Iron Phosphate (LFP) battery under strict "green charging" constraints.

By evaluating these methods on 2022-2024 ERCOT market data, we investigate the trade-offs between online deterministic planning and offline Reinforcement Learning.

## Project Overview

As renewable penetration increases in Texas, the intermittency of wind and solar introduces volatility and arbitrage opportunities. This project simulates a BESS that provides additive value to the grid by recapturing wasted renewable energy.

### Key Constraints & System Specs
* **Market:** ERCOT (Houston, North, South, West Zones).
* **Battery Capacity:** 100 MW Power / 400 MWh Energy (4-hour duration).
* **Green Charging Constraint:** The battery may only charge using renewable generation that is curtailed or in surplus (defined as generation exceeding 25% of load zone demand).
* **Objective:** Maximize arbitrage profit subject to degradation costs and physical constraints.

## Methodologies

This project compares two distinct approaches to the BESS control problem:

### 1. Lookahead with Rollout (Online Planning)
A deterministic, greedy heuristic that selects actions based on short-term trajectory projections.
* **Strategy:** Estimates the action-value of candidate actions by simulating effects forward over a 24-hour horizon (H=24).
* **Policy:** If current prices are low relative to the forecast window (K=12), the agent charges; if high, it discharges.
* **Strengths:** Proved most effective in high-volatility zones (Houston, South, West), demonstrating that long-horizon visibility is critical for daily arbitrage cycles.

### 2. Multi-Model Fitted Q-Iteration (Offline RL)
A batch Reinforcement Learning (RL) method that learns a policy directly from historical transitions without requiring online simulation.
* **Algorithm:** Iteratively converts the Bellman optimality equation into a sequence of supervised regression problems.
* **Function Approximator:** Histogram-based Gradient Boosting Regression Trees.
* **State Space:** Includes State of Charge (SoC), Net Load, Cyclical Time, Day-Ahead (DA) Price, 8-hour Price Forecast, and Price Volatility.
* **Strengths:** Showed robustness in the North zone and demonstrated the potential of learned value functions to capture market dynamics with limited feature horizons.

## Data & Simulation

The simulation uses hourly historical data from the ERCOT database spanning January 2022 to December 2024.

* **Training Set:** Jan 2022 - Sep 2024 (24,096 observations).
* **Testing Set:** Oct 2024 - Dec 2024 (2,208 observations).
* **Block Bootstrap Sampling:** To generate the training batch, we employed a Block Bootstrap Sampler (block length = 24 hours) to preserve intra-day autocorrelation while generating synthetic Real-Time price trajectories from Day-Ahead forecasts.

## Results

We benchmarked our models against a "Naive" heuristic (which charges when DA Price < $25 and discharges when DA Price > $100).

**Financial Performance Comparison (Oct - Dec 2024):**

| Model | Houston | North | South | West |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Benchmark** | $117,881 | $152,005 | $207,535 | $748,807 |
| **Fitted Q-Iteration (FQI)**| $364,509 | **$528,162** | $598,196 | $1,285,319 |
| **Lookahead (H=24)** | **$463,486** | $428,131 | **$751,913** | **$1,427,450** |

* **Lookahead** achieved the highest profits in 3 out of 4 zones, capitalizing on extreme price spreads in the West and South.
* **FQI** outperformed the Naive benchmark significantly in all zones and beat the Lookahead model in the North zone, despite using a shorter 8-hour forecast horizon.

## Requirements

* Python 3.8+
* scikit-learn (for GradientBoostingRegressor)
* pandas / numpy (for data manipulation)
* matplotlib (for visualization)


## Reference
This code accompanies the report:
*PowerPlay: Optimal Energy Storage Decisions in ERCOT* (2024). Stanford University, Decision Making Under Uncertainty.
