import os
import pandas
import numpy
import datetime
import queue
import time
import pickle

from CONSTANTS_MAIN import YEAR_LENGTH, TRANSITION_PERIOD, WINDOW_SIZE, STOCKS, \
    PARAMS_FNAME, RETURNS_FNAME, NUM_WORLD_STATES

from world_state_utils import get_data, filter_df, get_returns_df, get_state_returns, \
    get_state_params

# Scheme 1:
#     1) S&P goes up, starts above 50 day moving avg
#     2) S&P goes up, starts below 50 day moving avg
#     3) S&P goes down, starts above 50 day moving avg
#     4) S&P goes down, starts below 50 day moving avg
def compute_world_states_returns():

    states_dates = list()
    # List of world state dates represented as (start,end) tuples
    for i in range(NUM_WORLD_STATES):
        states_dates.append(list())

    # Filter the df to get data between 2005-2018
    start = datetime.date(2005,1,1)
    end = datetime.date(2017,12,31)

    # Get S&P500 returns
    returns_df = get_returns_df(["GSPC"], start, end)

    dates = returns_df['Date'].values
    returns = returns_df['GSPC'].values

    # Now figure out where to add the dates (i.e. assign time periods to
    # world states

    # the number of days per period
    curr_index = 0
    iteration_num = 0

    all_data = list()

    while(True):
        #print("On iteration {}, date {}".format(iteration_num, dates[curr_index]))

        # if we don't have enough observations, break out
        if (curr_index + TRANSITION_PERIOD >= len(returns)):
            break

        # determine the world state
        total_ret = sum(returns[curr_index:curr_index+TRANSITION_PERIOD])
        first_date = dates[curr_index]
        last_date = dates[curr_index + TRANSITION_PERIOD - 1]

        all_data.append((total_ret, (first_date, last_date)))

        curr_index += TRANSITION_PERIOD

    def get_first_elem(elem):
        return elem[0]

    all_data = sorted(all_data, key=get_first_elem)

    num_per_bucket = int(len(all_data) / NUM_WORLD_STATES)

    curr_index = 0
    for i in range(NUM_WORLD_STATES):
        for j in range(num_per_bucket):
            states_dates[i].append(all_data[curr_index][1])
            curr_index += 1

    return states_dates




# helper, takes all params as args
def returns_driver_helper(params_fname, returns_fname):
    # get the dates for the states

    state_dates = compute_world_states_returns()

    # for each state:
    #     combine returns into a dataframe (per-state dataframe)
    #     compute the mean and cov matrices of those lists
    #     this gives us mean vector and covariance matrix for each state

    # get returns
    stocks = STOCKS
    start = datetime.date(2005,1,1)
    end = datetime.date(2017,12,31)
    returns_df = get_returns_df(stocks, start, end)

    # get returns dataframes on a per-state basis
    state_returns = list()
    for state_date in state_dates:
        curr_st_rets = get_state_returns(state_date, returns_df)
        curr_st_rets = curr_st_rets[stocks]
        state_returns.append(numpy.reshape(curr_st_rets.values, (len(curr_st_rets),len(stocks))))

    # get mean vectors and covariance matrix
    means, sigmas = get_state_params(state_returns)

    parameters = (means, sigmas)

    pickle_out = open(params_fname, 'wb')
    pickle.dump(parameters, pickle_out)
    pickle_out.close()

    pickle_out = open(returns_fname, 'wb')
    pickle.dump(state_returns, pickle_out)
    pickle_out.close()



def world_returns_driver():


    t0 = time.time()
    returns_driver_helper(params_fname=PARAMS_FNAME,
                          returns_fname=RETURNS_FNAME)
    t1 = time.time()

    print("Time: {}".format(t1-t0))

world_returns_driver()

# Filenames:
#     'scheme1_1.pickle: means, sigmas for 4-state model with S&P up,
#                        starting above/below
