from numpy import array

# number of trading days in a year
# this is what annualized volatility is based on
YEAR_LENGTH = 252

# Valid transitions between world states
WORLD_STATE_TRANSITION = array([[1, 0, 1, 0],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [0, 1, 0, 1]])

TRANSITION_PERIOD = 20

NUMBER_OF_PERIODS = 100

# Number of world states
NUM_WORLD_STATES = 4

NUM_TRAIN_ITER = 10

SUCCESS_THRESHOLD = 0