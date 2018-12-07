import numpy
import math
import os
import pandas
import numpy
import datetime
import matplotlib.pyplot as plt
import random
import queue
import time
import pickle
import copy

from CONSTANTS_MAIN import YEAR_LENGTH, TRANSITION_PERIOD, PARAMS_FNAME, RETURNS_FNAME, \
    PORTFOLIO_FNAME, TRADER_WORLD_STATE_TRANSITION, RISK_FREE_RATE

# simulate correlated gbms
def simulate(mean, sigma, initial, raw_prices_dict):

    # get the cholesky factorization
    cholesky = numpy.linalg.cholesky(sigma)

    curr_prices = copy.deepcopy(initial)
    prices_dict = copy.deepcopy(raw_prices_dict)

    num_stocks = len(mean)
    t = float(1.0) / float(YEAR_LENGTH)
    sqrt_t = math.sqrt(t)

    # prices dictionary
    if prices_dict is None:
        prices_dict = dict()
        for i in range(num_stocks):
            prices_dict[i] = list()

    # iterate and simulate the correlated GBMs
    for i in range(TRANSITION_PERIOD):

        # get the random vector for this time step
        zs = numpy.random.normal(0.0, 1.0, num_stocks)

        for j in range(num_stocks):

            # get the current price of the stock
            curr_price = curr_prices[j]
            curr_mean = mean[j]
            curr_sigma_sq = sigma[j,j]

            # get row i of the cholesky matrix
            curr_row = cholesky[j,:]

            # compute the thing in the exponent
            exponent = (sqrt_t * (numpy.dot(curr_row, zs))) + ((curr_mean - (0.5*curr_sigma_sq))*t)

            new_price = curr_price * math.exp(exponent)

            curr_prices[j] = new_price
            prices_dict[j].append(new_price)

    return curr_prices, prices_dict

def simulate_driver_scheme1(state_num, init_prices, num_iterations,
                            mean, sigma, prices_dict):
    '''
    :param state_num: State number
    :param prices: Initial prices
    :param num_iterations: Number of realizations of GBM to average over
    :return:
    '''

    init_local = copy.deepcopy(init_prices)

    new_prices = numpy.zeros(len(init_prices))

    for i in range(num_iterations):
        curr_prices, prices_dict = simulate(mean, sigma, init_local, prices_dict)

        new_prices = numpy.add(curr_prices, new_prices)

    avg_prices = numpy.divide(new_prices, num_iterations)

    # don't average over path, but this is fine - we don't need this functionality

    return avg_prices, prices_dict


def simulate_driver(transitions):

    # get parameters for each world state
    pickle_in = open(PARAMS_FNAME, 'rb')
    params = pickle.load(pickle_in)
    pickle_in.close()

    means = params[0]
    sigmas = params[1]

    # get optimal portfolios
    pickle_in = open(PORTFOLIO_FNAME, 'rb')
    portfolios = pickle.load(pickle_in)
    pickle_in.close()

    real_ports = portfolios[0]
    optimal_ports = portfolios[1]

    prev_state = None
    curr_prices = [100.0] * len(means[0])

    # initialize prices dictionary
    prices_dict = dict()
    long_prices = dict()
    returns_dict = dict()

    num_stocks = len(means[0])

    for i in range(num_stocks):
        prices_dict[i] = list()
        long_prices[i] = list()
        long_prices[i].append(100.0)
        returns_dict[i] = list()


    portfolio_returns = list()

    for transition in transitions:
        # initialize prev_state
        if prev_state is None:
            prev_state = transition
            prev_prices = copy.deepcopy(curr_prices)
            continue

        # prev_state is the state that is ending right now
        # transition is the state that is coming up that we have to simulate for
        curr_portfolio = real_ports[prev_state]
        mean = means[transition]
        sigma = sigmas[transition]

        curr_prices, prices_dict = simulate_driver_scheme1(transition, curr_prices,
                                                           num_iterations=3,
                                                           mean=mean, sigma=sigma,
                                                           prices_dict=prices_dict)

        curr_returns = list()

        for i in range(num_stocks):
            long_prices[i].append(curr_prices[i])
            curr_stock_ret = math.log(curr_prices[i]) - math.log(prev_prices[i])
            returns_dict[i].append(curr_stock_ret)
            curr_returns.append(curr_stock_ret)

        curr_returns = numpy.array(curr_returns)
        port_return = numpy.dot(curr_portfolio, curr_returns) + (1-numpy.sum(curr_portfolio))*RISK_FREE_RATE*(float(TRANSITION_PERIOD) / float(YEAR_LENGTH))
        portfolio_returns.append(port_return)
        prev_prices = curr_prices

    portfolio_returns = numpy.array(portfolio_returns)

    avg_return = numpy.mean(portfolio_returns) * (float(YEAR_LENGTH) / float(TRANSITION_PERIOD))
    risk = numpy.std(portfolio_returns) * math.sqrt(float(YEAR_LENGTH) / float(TRANSITION_PERIOD))

    sharpe = (avg_return - RISK_FREE_RATE) / risk

    return avg_return, risk, sharpe

