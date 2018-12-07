from numpy import array

# number of trading days in a year
# this is what annualized volatility is based on
YEAR_LENGTH = 252

# Valid transitions between world states
WORLD_STATE_TRANSITION = array([[1, 0, 1, 0],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [0, 1, 0, 1]])

# Valid transitions between world states with trader-estimated
# intuition-based transition probabilities
TRADER_WORLD_STATE_TRANSITION = array([[0.7, 0, 0.3, 0],
                                [0.2, 0.4, 0.1, 0.3],
                                [0.3, 0.1, 0.4, 0.2],
                                [0, 0.4, 0, 0.6]])

TRANSITION_PERIOD = 20
WINDOW_SIZE = 50

NUMBER_OF_PERIODS = 100

# Number of world states
NUM_WORLD_STATES = 4
RF_INVESTMENT = [0, 0, 1.6, 1.6]

NUM_TRAIN_ITER = 5

SUCCESS_THRESHOLD = 0

STOCKS = ["AAPL", "GOOG", "MSFT"]

RISK_FREE_RATE = 0.02

STRATEGY = 'short_down'
PARAMS_FNAME = STRATEGY + '_params.pickle'
RETURNS_FNAME = STRATEGY + '_returns.pickle'
PORTFOLIO_FNAME = STRATEGY + '_portfolios.pickle'

# Scheme 1:
#     AAPL, GOOGL, MSFT
#     RF_INVESTMENT = [0, 0.1, 0.2, 0.3]
#     Mean-variance optimal trading strat

# Buy and Hold:
#     AAPL, GOOGL, MSFT
#     RF_INVESTMENT = [0, 0.1, 0.2, 0.3]
#     Simple buy and hold strategy

# Short when market is down
#     AAPL, GOOGL, MSFT
#     RF_INVESTMENT = [0, 0, 1.6, 1.6]
#     (0.07568668859234236, 0.20774170511511167, 0.26805733861424613)
#     (return, std. dev, sharpe ratio)
#
#     suggests some degree of mean reversion!!!