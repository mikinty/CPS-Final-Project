import math
import numpy
import datetime
import pickle

from CONSTANTS_MAIN import *

#from ws_4_up_below import compute_world_states_scheme1
from ws_returns import compute_world_states_returns
from world_state_utils import get_returns_df, filter_df, get_data

from dynamic_strats import mvo_optimal_strat, buy_and_hold_strat, long_short_strat

# compare function to sort state dates
def sort_state_date_cmp(a):
    return a[1][0]


def backtest_scheme1_helper(state_dates, returns_df, port_func):

    pickle_in = open(STRATEGY + '_returns.pickle', 'rb')
    state_returns = pickle.load(pickle_in)
    pickle_in.close()

    # reformat state dates into more friendly representation
    rf_state_dates = list()
    for i in range(len(state_dates)):
        rf_state_dates += [(i, state_date) for state_date in state_dates[i]]

    # sort them
    rf_state_dates = sorted(rf_state_dates, key=sort_state_date_cmp)

    # iterate and compute returns
    returns = list()
    prev_state = None

    ns = 1e-9  # number of seconds in a nanosecond

    for sd in rf_state_dates:

        if prev_state is None:
            prev_state = sd[0]
            continue

        start = sd[1][0].astype(int)
        start = datetime.datetime.utcfromtimestamp(start * ns)
        end = sd[1][1].astype(int)
        end = datetime.datetime.utcfromtimestamp(end * ns)

        curr_port = port_func(returns_df,
                              max_date=start-datetime.timedelta(days=1),
                              curr_state_end=prev_state, prev_port=None)

        if curr_port is None:
            prev_state = sd[0]
            continue

        curr_df = filter_df(returns_df, start, end)
        curr_df = curr_df[STOCKS]

        s = curr_df.sum(axis=0)
        rets = s.values

        port_ret = numpy.dot(curr_port, rets) + (1-numpy.sum(curr_port)) * RISK_FREE_RATE * (
                                  len(curr_df) / float(YEAR_LENGTH))
        returns.append(port_ret)

        prev_state = sd[0]

    returns = numpy.array(returns)
    avg_ret = numpy.mean(returns) * (float(YEAR_LENGTH) / float(TRANSITION_PERIOD))
    sd_ret = numpy.std(returns) * math.sqrt(float(YEAR_LENGTH) / float(TRANSITION_PERIOD))

    sharpe = (avg_ret - RISK_FREE_RATE) / sd_ret
    return avg_ret, sd_ret, sharpe


def backtest(world_states_func, port_func):

    # get the state dates
    state_dates = world_states_func()

    # get returns
    stocks = STOCKS
    start = datetime.date(2005,1,1)
    end = datetime.date(2017,12,31)
    returns_df = get_returns_df(stocks, start, end)

    return backtest_scheme1_helper(state_dates, returns_df, port_func)


print(backtest(compute_world_states_returns, long_short_strat))