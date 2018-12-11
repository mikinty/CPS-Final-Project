import math
import numpy
import pickle
import copy
import pandas

from CONSTANTS_MAIN import *

from dynamic_strats import mvo_optimal_strat, buy_and_hold_strat, long_short_strat



# simulate correlated gbms
def simulate(mean, sigma, initial, returns_df):
    # get the cholesky factorization
    cholesky = numpy.linalg.cholesky(sigma)

    curr_prices = copy.deepcopy(initial)

    num_stocks = len(mean)
    t = float(1.0) / float(YEAR_LENGTH)
    sqrt_t = math.sqrt(t)

    # iterate and simulate the correlated GBMs
    for i in range(TRANSITION_PERIOD):

        # get the random vector for this time step
        zs = numpy.random.normal(0.0, 1.0, num_stocks)

        returns = list()
        for j in range(num_stocks):
            # get the current price of the stock
            curr_price = curr_prices[j]
            curr_mean = mean[j]
            curr_sigma_sq = sigma[j, j]

            # get row i of the cholesky matrix
            curr_row = cholesky[j, :]

            # compute the thing in the exponent
            exponent = (sqrt_t * (numpy.dot(curr_row, zs))) + (
                        (curr_mean - (0.5 * curr_sigma_sq)) * t)

            new_price = curr_price * math.exp(exponent)

            returns.append(math.log(new_price) - math.log(curr_price))

            curr_prices[j] = new_price

        returns_df.loc[len(returns_df)] = returns

    return curr_prices, returns_df


def simulate_driver_scheme1(state_num, init_prices, num_iterations,
                            mean, sigma, returns_df):
    '''
    :param state_num: State number
    :param prices: Initial prices
    :param num_iterations: Number of realizations of GBM to average over
    :return:
    '''

    init_local = copy.deepcopy(init_prices)

    new_prices = numpy.zeros(len(init_prices))

    for i in range(num_iterations):
        curr_prices, returns_df = simulate(mean, sigma, init_local, returns_df)

        new_prices = numpy.add(curr_prices, new_prices)

    avg_prices = numpy.divide(new_prices, num_iterations)

    # don't average over path, but this is fine - we don't need this functionality

    return avg_prices, returns_df


pickle_in = open(STRATEGY_BUY_HOLD + '_params.pickle', 'rb')
BH_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_SCHEME + '_params.pickle', 'rb')
SC_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_SHORT_DOWN + '_params.pickle', 'rb')
SD_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_MVO_RETURNS_WS + '_params.pickle', 'rb')
MVO_RETURNS_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_1 + '_params.pickle', 'rb')
STRAT1_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_2 + '_params.pickle', 'rb')
STRAT2_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_3 + '_params.pickle', 'rb')
STRAT3_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_4 + '_params.pickle', 'rb')
STRAT4_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_5 + '_params.pickle', 'rb')
STRAT5_params = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_6 + '_params.pickle', 'rb')
STRAT6_params = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(STRATEGY_BUY_HOLD + '_portfolios.pickle', 'rb')
BH_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_SCHEME + '_portfolios.pickle', 'rb')
SC_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_SHORT_DOWN + '_portfolios.pickle', 'rb')
SD_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_MVO_RETURNS_WS + '_portfolios.pickle', 'rb')
MVO_RETURNS_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_1 + '_portfolios.pickle', 'rb')
STRAT1_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_2 + '_portfolios.pickle', 'rb')
STRAT2_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_3 + '_portfolios.pickle', 'rb')
STRAT3_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_4 + '_portfolios.pickle', 'rb')
STRAT4_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_5 + '_portfolios.pickle', 'rb')
STRAT5_portfolios = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open(STRATEGY_6 + '_portfolios.pickle', 'rb')
STRAT6_portfolios = pickle.load(pickle_in)
pickle_in.close()


def simulate_driver(transitions, strat, port_func):
    numpy.random.seed()

    # get parameters for each world state
    if strat == STRATEGY_BUY_HOLD:
        params = BH_params
        portfolios = BH_portfolios
    elif strat == STRATEGY_SCHEME:
        params = SC_params
        portfolios = SC_portfolios
    elif strat == STRATEGY_SHORT_DOWN:
        params = SD_params
        portfolios = SD_portfolios
    elif strat == STRATEGY_MVO_RETURNS_WS:
        params = MVO_RETURNS_params
        portfolios = MVO_RETURNS_portfolios
    elif strat == STRATEGY_1:
        params = STRAT1_params
        portfolios = STRAT1_portfolios
    elif strat == STRATEGY_2:
        params = STRAT2_params
        portfolios = STRAT2_portfolios
    elif strat == STRATEGY_3:
        params = STRAT3_params
        portfolios = STRAT3_portfolios
    elif strat == STRATEGY_4:
        params = STRAT4_params
        portfolios = STRAT4_portfolios
    elif strat == STRATEGY_5:
        params = STRAT5_params
        portfolios = STRAT5_portfolios
    elif strat == STRATEGY_6:
        params = STRAT6_params
        portfolios = STRAT6_portfolios

    means = params[0]
    sigmas = params[1]

    prev_state = None
    curr_prices = [100.0] * len(means[0])

    num_stocks = len(means[0])

    portfolio_returns = list()

    debug = list()

    # intiialize returns df
    returns_df = dict()
    for i in range(num_stocks):
        returns_df[i] = list()
        returns_df = pandas.DataFrame(data=returns_df)

    curr_loc = 0

    curr_portfolio = None

    for transition in transitions:
        # initialize prev_state
        if prev_state is None:
            prev_state = transition
            prev_prices = copy.deepcopy(curr_prices)
            continue

        # prev_state is the state that is ending right now
        # transition is the state that is coming up that we have to simulate for
        #curr_portfolio = real_ports[prev_state]
        curr_portfolio = port_func(returns_df, max_date=None, curr_state_end=prev_state,
                                   prev_port=curr_portfolio)

        mean = means[transition]
        sigma = sigmas[transition]

        curr_prices, returns_df = simulate_driver_scheme1(transition, curr_prices,
                                              num_iterations=3,
                                              mean=mean, sigma=sigma,
                                                          returns_df=returns_df)

        curr_returns = list()

        for i in range(num_stocks):
            curr_stock_ret = math.log(curr_prices[i]) - math.log(prev_prices[i])
            #returns_df[i].append(curr_stock_ret)
            curr_returns.append(curr_stock_ret)

        #returns_df.loc[curr_loc] = curr_returns
        #curr_loc += 1

        # ignore if the current portfolio is nothing given
        if curr_portfolio is None:
            prev_state = transition
            prev_prices = copy.deepcopy(curr_prices)
            continue

        curr_returns = numpy.array(curr_returns)
        port_return = numpy.dot(curr_portfolio, curr_returns) + (
                    1 - numpy.sum(curr_portfolio)) * RISK_FREE_RATE * (
                                  float(TRANSITION_PERIOD) / float(YEAR_LENGTH))
        debug.append((transition, port_return))
        portfolio_returns.append(port_return)
        prev_prices = curr_prices

        # remember your previous state
        prev_state = transition

    portfolio_returns = numpy.array(portfolio_returns)
    #print(portfolio_returns)
    avg_return = numpy.mean(portfolio_returns) * (
                float(YEAR_LENGTH) / float(TRANSITION_PERIOD))
    risk = numpy.std(portfolio_returns) * math.sqrt(
        float(YEAR_LENGTH) / float(TRANSITION_PERIOD))

    sharpe = (avg_return - RISK_FREE_RATE) / risk
    if (numpy.isnan(sharpe)):
        print(risk)
        sharpe = 0.0

    return avg_return, risk, sharpe

# x = 0
# y = 0
# z = 0
# for i in range(10):
#     a = simulate_driver([0,1,2,1,2,3,2,3,4,5,6,7,6,4,5,4,5,6,5,4,3,2,4,3,4,3,3,3,3,2,3], STRATEGY, long_short_strat)
#     x += a[0]
#     y += a[1]
#     z += a[2]
#
# print(x / 10.0)
# print(y / 10.0)
# print(z / 10.0)
