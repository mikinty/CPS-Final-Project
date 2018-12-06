# File Directory

This directory contains all the code for generating simulations, 
training our adversarial scheduler, and producing results for our project.

Some files:

- `RL/`: Contains files related to the Reinforcement Learning part of the project.
         Implements a scheduler optimizer that is suggested by Henriques et al.
  - `RL/CONSTANTS_RL.py`: Contains definitions for the Reinforcement Learning part of the project
  - `RL/schedOpt.py`: Optimizes the adversarial scheduler that tries to make the trading strategy lose money
- `strats/`: Contains the trading strategies to be tested