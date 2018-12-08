import numpy
import pickle

from CONSTANTS_MAIN import *

# gets optimal portfolio assuming knowledge of state, for next potential states
def save_hand_opt_portfolios():


    optimal_ports = dict()
    real_ports = dict()

    allocations = list()

    # Fill in allocations


    uni_port = [1.0 / float(len(STOCKS))] * len(STOCKS)
    uni_port = numpy.array(uni_port)

    for state_num in range(NUM_WORLD_STATES):
        curr = numpy.multiply(allocations[i], (1-RF_INVESTMENT[state_num]))
        optimal_ports[state_num] = curr

    for state_num in range(NUM_WORLD_STATES):

        # get indices of possible transitions
        curr_row = WORLD_STATE_TRANSITION[state_num,:]
        ts = TRADER_WORLD_STATE_TRANSITION

        curr_portfolio = numpy.zeros(len(STOCKS))
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


save_hand_opt_portfolios()

# Computed statistics for buy and hold strategy
# Mean return: 0.1762
# Std. Deviation: 0.2118
# Sharpe: 0.7377

