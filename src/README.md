# Requirements

All code in this project assumes the use of Python 3. Some of the packages that are required 

# Usage

There is a 3 step process to using the code in this directory:

1. `python initScheduler scheduler_name`: Initializes a uniform transition matrix with the name `scheduler_name` under the directory `schedulers/`

2. `python train.py trading_strat scheduler_name`: Performs reinforcement learning with the specified `trading_strat`, which can be one of

```
'buy_hold', 'scheme1', 'short_down'
```

and a `scheduler_name` scheduler. You can adjust the parameters for the training in `CONSTANTS_MAIN.py` and `RL/CONSTANTS_RL.py`.


3. `python smc.py trading_strat scheduler_name`: Performs SMC (just Monte Carlo sampling basically) with the specified `trading_strat` and `scheduler_name`, reporting the percent that the scheduler wins.


# File Directory

This directory contains all the code for generating simulations, 
training our adversarial scheduler, and producing results for our project.

Some files:

- `RL/`: Contains files related to the Reinforcement Learning part of the project.
         Implements a scheduler optimizer that is suggested by Henriques et al.
  - `RL/CONSTANTS_RL.py`: Contains definitions for the Reinforcement Learning part of the project
  - `RL/schedOpt.py`: Optimizes the adversarial scheduler that tries to make the trading strategy lose money
- `strats/`: Contains the trading strategies to be tested
