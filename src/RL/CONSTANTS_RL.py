'''
Constants for the RL alg

Michael You
Abhishek Barghava
'''

from numpy import array

# How many epochs to train the scheduler
SCHEDULER_OPTIMIZE_ITER = 100

# Number of iterations to train the scheduler
SCHEDULER_TRAIN_ITER = 50

# How much exploration we do during RL
EXPLORATION_RATE = 0

# How much exploration we do during move generation
EXPLORATION_CHOICE = 0

# How fast we are learning new choices
LEARNING_RATE = 0.1

# Number of world states
NUM_WORLD_STATES = 4

# Length of longest path to determine success/fail
MAX_SAMPLES = 365

EPSILON = 1e-50

CONVERGED_Q = array([[1, 0, 1, 0],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [0, 1, 0, 1]])
