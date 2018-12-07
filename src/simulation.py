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


# simulate correlated gbms
def simulate(mean, sigma, time_pd, initial):

    # get the cholesky factorization
    cholesky = numpy.linalg.cholesky(sigma)

    curr_prices = copy.deepcopy(initial)

    num_stocks = len(mean)
    t = float(1.0) / float(YEAR_LENGTH)
    sqrt_t = math.sqrt(t)

    # iterate and simulate the correlated GBMs
    for i in range(time_pd):

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

    return curr_prices

def simulate_driver(transition_period, fname, state_num, init_prices,
                    num_iterations):
    '''
    :param transition_period: Number of days per period
    :param fname: Filename to load params from
    :param state_num: State number
    :param prices: Initial prices
    :param num_iterations:
    :return:
    '''
    pickle_in = open(fname, 'rb')
    params = pickle.load(pickle_in)
    pickle_in.close()

    means = params[0]
    sigmas = params[1]

    new_prices = numpy.zeros(init_prices.shape)

    for i in range(num_iterations):
        curr_prices = simulate(means[state_num], sigmas[state_num],
                               transition_period, init_prices)

        new_prices = numpy.add(curr_prices, new_prices)


    avg_prices = numpy.divide(new_prices, num_iterations)
    return avg_prices
