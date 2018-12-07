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
import cvxopt
from cvxopt import blas, solvers

from .world_states import get_returns_df


def optimal_portfolio(returns):
    n = len(returns)
    returns = numpy.asmatrix(returns)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxcvxopt matrices
    S = cvxopt.matrix(numpy.cov(returns))
    pbar = cvxopt.matrix(numpy.mean(returns, axis=1))

    # Create constraint matrices
    G = -cvxopt.matrix(numpy.eye(n))  # negative n x n identity matrix
    h = cvxopt.matrix(0.0, (n, 1))
    A = cvxopt.matrix(1.0, (1, n))
    b = cvxopt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [numpy.sqrt(blas.dot(x, S * x)) for x in portfolios]

    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = numpy.polyfit(returns, risks, 2)
    x1 = numpy.sqrt(m1[2] / m1[0])

    # CALCULATE THE cvxoptIMAL PORTFOLIO
    wt = solvers.qp(cvxopt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return numpy.asarray(wt), returns, risks

def optimal_portfolio_for_state(mean, sigma):



# gets optimal portfolio assuming knowledge of state, for next potential states
def get_optimal_portfolio(state_num, fname):

    pickle_in = open(fname, 'rb')
    params = pickle.load(pickle_in)
    pickle_in.close()

    means = params[0]
    sigmas = params[1]

    # get mean and sigma
    mean = means[state_num]
    sigma = sigmas[state_num]

    # get indices of possible transitions


    #weights, returns, risks = cvxoptimal_portfolio(return_vec)

    #plt.plot(stds, means, 'o')
    #plt.ylabel('mean')
    #plt.xlabel('std')
    #plt.plot(risks, returns, 'y-o')
