# MPC-RRT-HEDAC Exploration
Autonomous navigation in cluttered, partially observable environments using model predictive control (MPC), rapidly-exploring random tree (RRT), and heat equation driven area coverage (HEDAC).

## Overview
This project implements a hybrid exploration framework combining:
- Heat-based exploration (HEDAC)
- Local RRT planning
- MPC trajectory tracking with corridor constraints

## How to run
pip install -r requirements.txt
python main.py

## Structure
control/     -> MPC + dynamics
planning/    -> RRT + corridor generation
environment/ -> heat + obstacles
utils/       -> geometry tools

## Reproducibility
Set seed in main.py for deterministic runs.

## Results
Example outputs in /results
