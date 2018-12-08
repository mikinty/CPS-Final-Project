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