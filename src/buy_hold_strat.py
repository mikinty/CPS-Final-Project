import numpy
import pickle

from CONSTANTS_MAIN import STOCKS, NUM_WORLD_STATES, PORTFOLIO_FNAME, RF_INVESTMENT

# gets optimal portfolio assuming knowledge of state, for next potential states
def save_buy_hold_portfolios():


    optimal_ports = dict()
    real_ports = dict()


    uni_port = [1.0 / float(len(STOCKS))] * len(STOCKS)
    uni_port = numpy.array(uni_port)

    for state_num in range(NUM_WORLD_STATES):
        curr = numpy.multiply(uni_port, (1-RF_INVESTMENT[state_num]))
        optimal_ports[state_num] = curr
        real_ports[state_num] = curr

    opts = (real_ports, optimal_ports)

    # save
    pickle_out = open(PORTFOLIO_FNAME, 'wb')
    pickle.dump(opts, pickle_out)
    pickle_out.close()


save_buy_hold_portfolios()

# Computed statistics for buy and hold strategy
# Mean return: 0.1762
# Std. Deviation: 0.2118
# Sharpe: 0.7377

