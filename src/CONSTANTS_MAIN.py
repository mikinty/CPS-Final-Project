from numpy import array
import datetime

# number of trading days in a year
# this is what annualized volatility is based on
YEAR_LENGTH = 252

# Valid transitions between world states
ORIG_TRANSITION = array([[1, 0, 1, 0],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [0, 1, 0, 1]])

# Valid transitions between world states with trader-estimated
# intuition-based transition probabilities
FOUR_TRADER_WORLD_STATE_TRANSITION = array([[0.7, 0, 0.3, 0],
                                            [0.2, 0.4, 0.1, 0.3],
                                            [0.3, 0.1, 0.4, 0.2],
                                            [0, 0.4, 0, 0.6]])

EIGHT_TRADER_WORLD_STATE_TRANSITION = array([[0.3,0.7,0,0,0,0,0,0],
                                             [0.2,0.3,0.5,0,0,0,0,0],
                                             [0,0.5,0.3,0.2,0,0,0,0],
                                             [0,0,0.2,0.3,0.2,0,0,0],
                                             [0,0,0,0.5,0.3,0.2,0,0],
                                             [0,0,0,0,0.5,0.3,0.2,0],
                                             [0,0,0,0,0,0.5,0.3,0.2],
                                             [0,0,0,0,0,0,0.7,0.3]])

weight_portfolio = False


# Valid transitions between world states
RET_WORLD_STATE_TRANSITION = array([[1,1,0,0,0,0,0,0],
                                    [1,1,1,0,0,0,0,0],
                                    [0,1,1,1,0,0,0,0],
                                    [0,0,1,1,1,0,0,0],
                                    [0,0,0,1,1,1,0,0],
                                    [0,0,0,0,1,1,1,0],
                                    [0,0,0,0,0,1,1,1],
                                    [0,0,0,0,0,0,1,1]])


TRANSITION_PERIOD = 20
WINDOW_SIZE = 50

# how many decisions we make per run
NUMBER_OF_PERIODS = 100

# Number of training epochs
NUM_TRAIN_ITER = 20

# sharpe ratio required as a win
SUCCESS_THRESHOLD = 0


STRATEGY_BUY_HOLD = 'buy_hold'
STRATEGY_SCHEME = 'scheme1' # MVO, 4 world states
STRATEGY_SHORT_DOWN = 'short_down'
STRATEGY_MVO_RETURNS_WS = 'mvo_returns' # Mvo, 8 world states
STRATEGY_1 = 'strat1'
STRATEGY_2 = 'strat2'
STRATEGY_3 = 'strat3'
STRATEGY_4 = 'strat4'
STRATEGY_5 = 'strat5'
STRATEGY_6 = 'strat6'
STRATEGY_7 = 'strat7'
STRATEGY_8 = 'strat8'

# These parameters define tweakable trading strategy properties
START_DATE = datetime.date(2005,1,1)
END_DATE = datetime.date(2018,12,3)
WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
NUM_WORLD_STATES = 8
RF_INVESTMENT = [0, 0, 0, 0, 0, 0, 0, 0]
STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
RISK_FREE_RATE = 0.02
STRATEGY = STRATEGY_8
TRADER_WORLD_STATE_TRANSITION = EIGHT_TRADER_WORLD_STATE_TRANSITION



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

# ID: 1
# Name: strat1
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [0.0]*8
# STOCKS = ["AAPL", "GOOG", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_1
# TRADER_WORLD_STATE_TRANSITION = None
# Using MVO

# ID: 2
# Name: strat2
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [0.0]*8
# STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_2
# TRADER_WORLD_STATE_TRANSITION = None
# Using MVO

# ID: 3
# Name: strat3
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [1.5, 1.5, 0.5, 0, 0, 0, 0, 0]
# STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_3
# TRADER_WORLD_STATE_TRANSITION = None
# Using hand opt

# ID: 4
# Name: strat4
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [1, 1.5, 0.5, 0, 0, 0, 0, 0]
# STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_3
# TRADER_WORLD_STATE_TRANSITION = None
# Using hand opt

# ID: 5
# Name: strat5
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [1, 1.5, 0.5, 0, 0, 0, 0, 0]
# STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_3
# TRADER_WORLD_STATE_TRANSITION = EIGHT_WORLD_TRANS
# Using hand opt

# ID: 6
# Name: strat6
# WORLD_STATE_TRANSITION = RET_WORLD_STATE_TRANSITION
# NUM_WORLD_STATES = 8
# RF_INVESTMENT = [1, 1.0, 1.0, 0, 0, 0, 0, 0]
# STOCKS = ["AAPL", "F", "JNJ", "JPM", "XOM"]
# RISK_FREE_RATE = 0.02
# STRATEGY = STRATEGY_3
# TRADER_WORLD_STATE_TRANSITION = EIGHT_WORLD_TRANS
# Using hand opt
