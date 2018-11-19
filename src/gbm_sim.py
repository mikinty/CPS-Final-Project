import numpy
import math
import os
import pandas
import numpy
import datetime
import matplotlib.pyplot as plt
import random

# data file
DATA_FILE = os.path.expanduser("~/College/Junior/15424_Foundations/Project/data/")

# number of trading days in a year
# this is what annualized volatility is based on
YEAR_LENGTH = 252

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



# simulate geometric Brownian motion
# TODO: Adjust this to work with arbitrary number of stocks
def gbm_sim(stock1, stock2, years, ticker1, ticker2):

    # iterate over the years
    for year in years:
        # get the year-specific dataframes
        s1_y = stock1[(stock1['Date'] >= datetime.date(year, 1, 1))
                          & (stock1['Date'] <= datetime.date(year, 12, 31))]
        s2_y = stock2[(stock2['Date'] >= datetime.date(year, 1, 1))
                          & (stock2['Date'] <= datetime.date(year, 12, 31))]

        # get the adjusted close stock prices
        adj_close1 = s1_y['Adj Close'].values
        adj_close2 = s2_y['Adj Close'].values

        num_obs1 = len(adj_close1)
        num_obs2 = len(adj_close2)
        assert(num_obs1 == num_obs2)

        # compute the mean and annualized variance and correlation coefficient
        diff1 = numpy.subtract(numpy.log(adj_close1[1:]), numpy.log(adj_close1[:-1]))
        mu1 = numpy.mean(diff1) * num_obs1
        sigma1 = numpy.std(diff1) * math.sqrt(num_obs1)

        diff2 = numpy.subtract(numpy.log(adj_close2[1:]), numpy.log(adj_close2[:-1]))
        mu2 = numpy.mean(diff2) * num_obs2
        sigma2 = numpy.std(diff2) * math.sqrt(num_obs2)

        corr = numpy.corrcoef(diff1, diff2)[0,1]

        # random seeds
        random.seed(174)
        numpy.random.seed(175)

        # simulate the geometric Brownian motions
        gbm1 = list()
        gbm2 = list()
        gbm1.append(adj_close1[0])
        gbm2.append(adj_close2[0])

        dates = s1_y['Date'].values

        # get lower triangular cholesky matrix
        cholesky_mat = numpy.zeros((2,2))
        cholesky_mat[0][0] = sigma1
        cholesky_mat[0][1] = 0
        cholesky_mat[1][0] = sigma2 * corr
        cholesky_mat[1][1] = sigma2 * math.sqrt(1 - (corr**2))

        # set time step
        time_step = 1.0 / float(num_obs1)

        # useful for computing the simulation
        sqrt_tstep = math.sqrt(time_step)
        #mu1_tstep = mu1 * time_step
        #mu2_tstep = mu2 * time_step
        mu1_tstep = (mu1 - (0.5 * (sigma1**2))) * time_step
        mu2_tstep = (mu2 - (0.5 * (sigma2**2))) * time_step


        # compute the next element of the GBM
        for i in range(len(dates)-1):
            z1 = numpy.random.standard_normal(1)
            z2 = numpy.random.standard_normal(1)

            gbm1_core = (sigma1 * z1)
            new_gbm1 = gbm1[i] * math.exp((sqrt_tstep * gbm1_core) + mu1_tstep)
            gbm1.append(new_gbm1)

            gbm2_core = (cholesky_mat[1][0] * z1) + (cholesky_mat[1][1] * z2)
            new_gbm2 = gbm2[i] * math.exp((sqrt_tstep * gbm2_core) + mu2_tstep)
            gbm2.append(new_gbm2)

        # Plot the stocks
        plt.clf()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Date")
        ax1.set_ylabel("{} Price".format(ticker1))
        ax1.plot(dates, adj_close1, label=ticker1, color='red')
        ax1.plot(dates, gbm1, label="Simulated {}".format(ticker1), color='blue')

        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("{} Price".format(ticker2))
        ax2.plot(dates, adj_close2, label=ticker2, color='green')
        ax2.plot(dates, gbm2, label="Simulated {}".format(ticker2), color='purple')


        # otherwise the right y-label is slightly clipped
        fig.tight_layout()

        # show legend
        fig.legend(loc=2)

        plt.show()



if __name__ == "__main__":
    # get data for Apple
    apple_df = get_data("AAPL")
    apple_dates = apple_df['Date'].values

    # get data for Google
    google_df = get_data("GOOG")
    google_dates = google_df['Date'].values

    # get intersection of dates and update dataframes
    dates = list(set(apple_dates).intersection(google_dates))
    apple_df = apple_df.loc[apple_df['Date'].isin(dates)]
    google_df = google_df.loc[google_df['Date'].isin(dates)]

    # set the years
    start_year = 2005
    end_year = 2017

    years = list(range(start_year, end_year+1, 1))

    # simulate geometric Brownian motion
    gbm_sim(apple_df, google_df, years, "AAPL", "GOOG")