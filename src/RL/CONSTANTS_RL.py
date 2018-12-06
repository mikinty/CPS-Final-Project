'''
Constants for the RL alg

Michael You
Abhishek Barghava
'''

# How many epochs to train the scheduler
SCHEDULER_OPTIMIZE_ITER = 100

# Number of iterations to train the scheduler
SCHEDULER_TRAIN_ITER = 1000

# How much exploration we do during RL
EXPLORATION_RATE = 0.05

# How fast we are learning new choices
LEARNING_RATE = 0.05

# Length of longest path to determine success/fail
MAX_SAMPLES = 365