
import numpy
import pickle
import cvxopt
from cvxopt import blas, solvers
import scipy.optimize as sco

from world_state_utils import is_df, filter_df

from CONSTANTS_MAIN import *


# This is based on theoretical closed form solution of the Lagrangian optimization
# for mean-variance optimal portfolios
# This also assumes zero-interest rate, which is fine because the trader can
# do anything he/she wants!
def optimal_portfolio_for_state(mean, sigma, rf_prop):
    alpha_star = numpy.matmul(numpy.linalg.inv(sigma), mean)
    alpha_star = numpy.divide(alpha_star, sum(alpha_star))
    alpha_star = numpy.multiply(alpha_star, (1.0-rf_prop))

    return alpha_star



def mvo_optimal_strat(returns_df, max_date, curr_state_end, prev_port):

    rf = RF_INVESTMENT
    rf_prop = 0.0
    ts = TRADER_WORLD_STATE_TRANSITION

    returns_df = filter_df(returns_df, None, max_date)

    if 'Date' in returns_df.columns:
        #print(returns_df)
        returns_df = returns_df.drop(['Date'], axis=1)

    trader_window = 60

    returns_df = returns_df.tail(trader_window)

    if (len(returns_df) < trader_window):
        return None

    returns_mat = returns_df.values
    mean_vec = numpy.mean(returns_mat, axis=0)
    sigma = numpy.cov(returns_mat, rowvar=False)

    if (curr_state_end == NUM_WORLD_STATES-1):
        rf_prop = (rf[curr_state_end] + rf[curr_state_end-1]) / 2.0
    elif (curr_state_end == 0):
        rf_prop = (rf[0] + rf[1]) / 2.0
    else:
        rf_prop = (rf[curr_state_end] + rf[curr_state_end+1] + rf[curr_state_end-1]) / 3.0

    port = optimal_portfolio_for_state(mean_vec, sigma, rf_prop)

    return port

def buy_and_hold_strat(returns_df, max_date, curr_state_end, prev_port):
    port = list()

    df = returns_df
    if 'Date' in returns_df.columns:
        df = df.drop(['Date'], axis=1)

    num_stocks = df.values.shape[1]
    for i in range(num_stocks):
        port.append(1.0 / float(num_stocks))

    return port

def long_short_strat_improved(returns_df, max_date, curr_state_end, prev_port):

    port = list()

    window_rev = 60
    window_mom = 20
    #window_rev = 60
    #window_mom = 30

    returns_df = filter_df(returns_df, None, max_date)

    if (len(returns_df) < window_rev):
        return None

    rev_returns_df = returns_df.tail(window_rev)
    mom_returns_df = returns_df.tail(window_mom)

    if 'Date' in returns_df.columns:
        rev_returns_df = rev_returns_df.drop(['Date'], axis=1)
        mom_returns_df = mom_returns_df.drop(['Date'], axis=1)

    rev_returns_mat = rev_returns_df.values
    mom_returns_mat = mom_returns_df.values

    num_stocks = rev_returns_mat.shape[1]

    rev_mean = numpy.mean(rev_returns_mat, axis=0)
    mom_mean = numpy.mean(mom_returns_mat, axis=0)

    #if (numpy.sum(rev_mean) < 0 and numpy.sum(mom_mean) > 0):
    #    for i in range(num_stocks):
    #        port.append(1.0 / float(num_stocks))
    #elif (numpy.sum(rev_mean) > 0 and numpy.sum(mom_mean) < 0):
    #    for i in range(num_stocks):
    #        port.append(-1.0 / float(num_stocks))
    #else:
    #    for i in range(num_stocks):
    #        port.append(0.0)
    if (curr_state_end <= 2):
        if curr_state_end == 2:
            for i in range(num_stocks):
                port.append(-0.2 / num_stocks)
        else:
            for i in range(num_stocks):
                port.append(-1.0 / num_stocks)
    elif (curr_state_end >= 5):
        for i in range(num_stocks):
            port.append(1.0 / num_stocks)
    else:
        for i in range(num_stocks):
            port.append(0.0)

    return numpy.array(port)


def long_short_strat(returns_df, max_date, curr_state_end, prev_port):

    port = list()

    window_rev = 60
    window_mom = 20
    #window_rev = 60
    #window_mom = 30

    returns_df = filter_df(returns_df, None, max_date)

    if (len(returns_df) < window_rev):
        return None

    rev_returns_df = returns_df.tail(window_rev)
    mom_returns_df = returns_df.tail(window_mom)

    if 'Date' in returns_df.columns:
        rev_returns_df = rev_returns_df.drop(['Date'], axis=1)
        mom_returns_df = mom_returns_df.drop(['Date'], axis=1)

    rev_returns_mat = rev_returns_df.values
    mom_returns_mat = mom_returns_df.values

    num_stocks = rev_returns_mat.shape[1]

    rev_mean = numpy.mean(rev_returns_mat, axis=0)
    mom_mean = numpy.mean(mom_returns_mat, axis=0)

    #if (numpy.sum(rev_mean) < 0 and numpy.sum(mom_mean) > 0):
    #    for i in range(num_stocks):
    #        port.append(1.0 / float(num_stocks))
    #elif (numpy.sum(rev_mean) > 0 and numpy.sum(mom_mean) < 0):
    #    for i in range(num_stocks):
    #        port.append(-1.0 / float(num_stocks))
    #else:
    #    for i in range(num_stocks):
    #        port.append(0.0)
    if (curr_state_end <= 2):
            for i in range(num_stocks):
                port.append(-1.0 / num_stocks)
    elif (curr_state_end >= 5):
        for i in range(num_stocks):
            port.append(1.0 / num_stocks)
    else:
        for i in range(num_stocks):
            port.append(0.0)

    return numpy.array(port)








