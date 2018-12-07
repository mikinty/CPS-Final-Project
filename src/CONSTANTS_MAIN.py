from numpy import array

# number of trading days in a year
# this is what annualized volatility is based on
YEAR_LENGTH = 252

# Valid transitions between world states
WORLD_STATE_TRANSITION = array([[1, 0, 1, 0],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [0, 1, 0, 1]])

# Valid transitions between world states
TRADER_WORLD_STATE_TRANSITION = array([[0.7, 0, 0.3, 0],
                                [0.2, 0.4, 0.1, 0.3],
                                [0.3, 0.1, 0.4, 0.2],
                                [0, 0.4, 0, 0.6]])

TRANSITION_PERIOD = 20
WINDOW_SIZE = 50

NUMBER_OF_PERIODS = 100

# Number of world states
NUM_WORLD_STATES = 4
RF_INVESTMENT = [0, 0.1, 0.2, 0.3]

NUM_TRAIN_ITER = 5

SUCCESS_THRESHOLD = 0

STOCKS = ["AAPL", "GOOG", "MSFT"]

RISK_FREE_RATE = 0.02

PARAMS_FNAME = 'scheme1_params.pickle'
RETURNS_FNAME = 'scheme1_returns.pickle'
PORTFOLIO_FNAME = 'scheme1_portfolios.pickle'