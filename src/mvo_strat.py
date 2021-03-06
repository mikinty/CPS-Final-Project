
import numpy
import pickle
import cvxopt
from cvxopt import blas, solvers
import scipy.optimize as sco

from CONSTANTS_MAIN import PARAMS_FNAME, PORTFOLIO_FNAME, NUM_WORLD_STATES, \
                WORLD_STATE_TRANSITION, RETURNS_FNAME, RISK_FREE_RATE, \
                RF_INVESTMENT, TRADER_WORLD_STATE_TRANSITION


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

# Taken from https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
# Doesn't work, but leaving it in for ideas
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = numpy.sum(mean_returns*weights ) * 252
    std = numpy.sqrt(numpy.dot(weights.T, numpy.dot(cov_matrix, weights))) * numpy.sqrt(252)
    return std, returns

# Taken from https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
# Doesn't work, but leaving it in for ideas
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

# Taken from https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
# Doesn't work, but leaving it in for ideas
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: numpy.sum(x) - 1})
    bound = (-1.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# This is based on theoretical closed form solution of the Lagrangian optimization
# for mean-variance optimal portfolios
# This also assumes zero-interest rate, which is fine because the trader can
# do anything he/she wants!
def optimal_portfolio_for_state(mean, sigma, rf_prop):
    alpha_star = numpy.matmul(numpy.linalg.inv(sigma), mean)
    alpha_star = numpy.divide(alpha_star, sum(alpha_star))
    alpha_star = numpy.multiply(alpha_star, (1.0-rf_prop))

    return alpha_star



# gets optimal portfolio assuming knowledge of state, for next potential states
def save_optimal_portfolios():


    pickle_in = open(PARAMS_FNAME, 'rb')
    params = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(RETURNS_FNAME, 'rb')
    returns = pickle.load(pickle_in)
    pickle_in.close()

    means = params[0]
    sigmas = params[1]

    optimal_ports = dict()
    real_ports = dict()

    rf = RF_INVESTMENT
    ts = TRADER_WORLD_STATE_TRANSITION

    for state_num in range(NUM_WORLD_STATES):
        optimal_ports[state_num] = optimal_portfolio_for_state(means[state_num],
                                                               sigmas[state_num],
                                                               rf[state_num])

    for state_num in range(NUM_WORLD_STATES):

        # get mean and sigma
        mean = means[state_num]
        sigma = sigmas[state_num]

        # get indices of possible transitions
        curr_row = WORLD_STATE_TRANSITION[state_num,:]

        curr_portfolio = numpy.zeros(len(mean))
        num_added = 0

        for i in range(len(curr_row)):
            # if we can't transition ignore this state
            if (curr_row[i] == 0):
                continue

            num_added += 1

            # This weighting of portfolios depends on my estimated probabilities of
            # transitioning from one state to another, which are solely based on
            # intuition.
            if ts is not None:
                curr_add = numpy.multiply(optimal_ports[i], ts[state_num, i])
            else:
                curr_add = optimal_ports[i]

            curr_portfolio = numpy.add(curr_portfolio, curr_add)

        if ts is None:
            curr_portfolio = numpy.divide(curr_portfolio, num_added)

        real_ports[state_num] = curr_portfolio

    opts = (real_ports, optimal_ports)

    # save
    pickle_out = open(PORTFOLIO_FNAME, 'wb')
    pickle.dump(opts, pickle_out)
    pickle_out.close()


save_optimal_portfolios()


# Computed statistics for mean-variance optimal strategy
# Mean return: 0.1609
# Std. Deviation: 0.2005
# Sharpe: 0.7028