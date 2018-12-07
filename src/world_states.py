import os
import pandas
import numpy
import datetime
import queue
import time
import pickle

from CONSTANTS_MAIN import YEAR_LENGTH, TRANSITION_PERIOD, WINDOW_SIZE, STOCKS, \
    PARAMS_FNAME, RETURNS_FNAME

# data file
DATA_FILE = os.path.expanduser("./data/")

# get data for specified ticker
def get_data(ticker):
    # the filename of the data
    fname = DATA_FILE + ticker + ".csv"

    # read data from file
    df = pandas.read_csv(fname)

    # convert date column to datetime
    df['Date'] = pandas.to_datetime(df['Date'])

    # return the resulting dataframe
    return df

# get returns for this df
def get_returns(df):
    adj_close = df['Adj Close'].values
    rets = numpy.subtract(numpy.log(adj_close[1:]),
                          numpy.log(adj_close[:-1]))
    dates = df['Date'].values

    # account for the first observation's return not being included
    dates = dates[1:]

    return dates, rets



def filter_df(df, start, end):
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    return df

# Scheme 1:
#     1) S&P goes up, starts above 50 day moving avg
#     2) S&P goes up, starts below 50 day moving avg
#     3) S&P goes down, starts above 50 day moving avg
#     4) S&P goes down, starts below 50 day moving avg
def compute_world_states_scheme1():
    # Get S&P data
    df = get_data("GSPC")

    # List of world state dates represented as (start,end) tuples
    st1 = list()
    st2 = list()
    st3 = list()
    st4 = list()

    # Filter the df to get data between 2005-2018
    start = datetime.date(2005,1,1)
    end = datetime.date(2017,12,31)
    df = filter_df(df, start, end)

    # Get dates and prices
    dates = df['Date'].values
    prices = df['Adj Close'].values

    # Compute 50 day rolling window
    curr_index = 0
    sum_window = 0
    curr_rolling_queue = queue.Queue()

    for i in range(WINDOW_SIZE):
        curr_rolling_queue.put(prices[curr_index])
        sum_window += prices[curr_index]
        curr_index += 1
    curr_avg = sum_window / float(WINDOW_SIZE)

    # Now figure out where to add the dates (i.e. assign time periods to
    # world states

    # the number of days per period
    iteration_num = 0
    while(True):
        #print("On iteration {}, date {}".format(iteration_num, dates[curr_index]))

        # if we don't have enough observations, break out
        if (curr_index + TRANSITION_PERIOD >= len(prices)):
            break

        # determine the world state
        first_price = prices[curr_index]
        last_price = prices[curr_index + TRANSITION_PERIOD - 1]

        # add the dates into the world state
        if (last_price > first_price):
            if (first_price > curr_avg):
                # goes up, starts above avg
                st1.append((dates[curr_index], dates[curr_index+TRANSITION_PERIOD-1]))
            else:
                st2.append((dates[curr_index], dates[curr_index+TRANSITION_PERIOD-1]))
        else:
            if (first_price > curr_avg):
                st3.append((dates[curr_index], dates[curr_index+TRANSITION_PERIOD-1]))
            else:
                st4.append((dates[curr_index], dates[curr_index+TRANSITION_PERIOD-1]))

        # now recompute the rolling average and update curr_index
        for j in range(TRANSITION_PERIOD):
            # remove the oldest price from the queue
            removed_price = float(curr_rolling_queue.get())
            # get the previous sum of prices in the queue
            prev_sum = curr_avg * WINDOW_SIZE
            # get the new sum by removing a price and adding the new price back in
            new_sum = prev_sum - removed_price + prices[curr_index]
            # put the new price into the queue
            curr_rolling_queue.put(prices[curr_index])
            # compute the new average
            curr_avg = new_sum / float(WINDOW_SIZE)
            # increment curr_index
            curr_index += 1
        iteration_num += 1

    return st1, st2, st3, st4

def get_returns_df(stocks, start, end):
    returns_dict = dict()
    for stock in stocks:
        # get the stock's dataframe
        stock_df = get_data(stock)
        stock_df = filter_df(stock_df, start, end)

        # get the returns
        dates, stock_rets = get_returns(stock_df)

        if (stock == stocks[0]):
            returns_dict["Date"] = dates

        returns_dict[stock] = stock_rets

    returns_df = pandas.DataFrame(data=returns_dict)
    return returns_df

# get state-specific returns
def get_state_returns(state_dates, returns_df):
    # loop over the date intervals
    df = pandas.DataFrame()

    for state_date in state_dates:
        # start and end dates
        start = state_date[0]
        end = state_date[1]

        # loop over the relevant dates
        curr_returns_df = filter_df(returns_df, start, end)

        # add to the dataframe
        if (state_date == state_dates[0]):
            df = curr_returns_df
        else:
            df = df.append(curr_returns_df, ignore_index=True)

    return df


# compute state parameters
def get_state_params(state_returns):

    means = list()
    sigmas = list()
    for state_return in state_returns:
        curr_mean = numpy.multiply(numpy.mean(state_return, axis=0),
                                   YEAR_LENGTH)
        sigma = numpy.multiply(numpy.cov(state_return, rowvar=False),
                                YEAR_LENGTH)
        means.append(curr_mean)
        sigmas.append(sigma)
    return means, sigmas


# helper, takes all params as args
def scheme1_driver_helper(params_fname, returns_fname):
    # get the dates for the states

    state_dates = compute_world_states_scheme1()

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



def scheme1_driver():


    t0 = time.time()
    scheme1_driver_helper(params_fname=PARAMS_FNAME,
                          returns_fname=RETURNS_FNAME)
    t1 = time.time()

    print("Time: {}".format(t1-t0))

scheme1_driver()
# Filenames:
#     'scheme1_1.pickle: means, sigmas for 4-state model with S&P up,
#                        starting above/below
