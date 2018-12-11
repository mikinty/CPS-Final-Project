import math
import numpy
import datetime
import pickle

from CONSTANTS_MAIN import *

#from ws_4_up_below import compute_world_states_scheme1
from ws_returns import compute_world_states_returns
from world_state_utils import get_returns_df, filter_df, get_data

from dynamic_strats import mvo_optimal_strat, buy_and_hold_strat, long_short_strat, long_short_strat_improved

import matplotlib.pyplot as plt

# compare function to sort state dates
def sort_state_date_cmp(a):
    return a[1][0]

def plot_many(dates, returns_list):


    prices_bh = [100.0]
    prices = [100.0]
    prices_imp = [100.0]

    returns_bh = returns_list[0]
    returns = returns_list[1]
    returns_imp = returns_list[2]

    curr_price_bh, curr_price, curr_price_imp = 100.0, 100.0, 100.0

    for i in range(len(returns)):
        curr_price_bh = curr_price_bh * math.exp(returns_bh[i])
        prices_bh.append(curr_price_bh)
        curr_price = curr_price * math.exp(returns[i])
        prices.append(curr_price)
        curr_price_imp = curr_price_imp * math.exp(returns_imp[i])
        prices_imp.append(curr_price_imp)

    #fig, ax = plt.subplots()

    #ax2 = ax.twinx()
    #ax3 = ax.twinx()

    #f, (ax1, ax2) = plt.subplots(1,2,sharex=False,sharey=False)
    plt.plot(dates, prices_bh, label="Buy and Hold")
    plt.plot(dates, prices, color="C2", label="Long Short")
    plt.plot(dates, prices_imp, color="C3", label="Long-Short Scheduler-Improved")

    plt.title('Strategy Backtest, 2005-2017')

    #ax.legend(loc=2)
    #ax2.legend(loc=4)
    #ax3.legend(loc=5)
    plt.legend(loc=2)
    plt.show()

def plot_returns(dates, returns):

    curr_price = 100.0

    prices = list()
    prices.append(curr_price)

    for i in range(len(returns)):
        curr_price = curr_price * math.exp(returns[i])
        prices.append(curr_price)

    ts = [i for i in range(len(prices))]

    plt.plot(dates, prices)
    plt.show()


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
    dates = returns_df['Date'].values
    prev_state = None

    ns = 1e-9  # number of seconds in a nanosecond
    dates = list()
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
        dates.append(start)
        returns.append(port_ret)

        prev_state = sd[0]

    returns = numpy.array(returns)

    dates.append(end)
    #plot_returns(dates, returns)

    avg_ret = numpy.mean(returns) * (float(YEAR_LENGTH) / float(TRANSITION_PERIOD))
    sd_ret = numpy.std(returns) * math.sqrt(float(YEAR_LENGTH) / float(TRANSITION_PERIOD))

    sharpe = (avg_ret - RISK_FREE_RATE) / sd_ret
    return avg_ret, sd_ret, sharpe, dates, returns


def backtest(world_states_func, port_func):

    # get the state dates
    state_dates = world_states_func()

    # get returns
    stocks = STOCKS
    start = datetime.date(2005,1,1)
    end = datetime.date(2017,12,31)
    returns_df = get_returns_df(stocks, start, end)

    return backtest_scheme1_helper(state_dates, returns_df, port_func)


#print(backtest(compute_world_states_returns, long_short_strat))
avg_ret, sd_ret, sharpe, dates_bh, returns_bh = backtest(compute_world_states_returns, buy_and_hold_strat)
avg_ret, sd_ret, sharpe, dates, returns = backtest(compute_world_states_returns, long_short_strat)
avg_ret, sd_ret, sharpe, dates_imp, returns_imp = backtest(compute_world_states_returns, long_short_strat_improved)

plot_many(dates, [returns_bh, returns, returns_imp])