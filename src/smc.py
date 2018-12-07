'''
Scheduler SMC

Michael You
Abhishek Barghava
'''

import pickle
import sys

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, NUM_WORLD_STATES

from numpy import array, zeros, random, divide, zeros
import multiprocessing as mp
from functools import partial

from simulation import simulate_driver
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Need 2 arguments: STRAT SCHEDULER_FILE')
        sys.exit()

    strat = sys.argv[1]
    SCHEDULER_FILE = './schedulers/'+ sys.argv[2] + '.pickle'

    print('Calculating success of', SCHEDULER_FILE)

    # Load scheduler file
    pickle_in = open(SCHEDULER_FILE, 'rb')
    scheduler = pickle.load(pickle_in)
    pickle_in.close()

    win = 0
    total = 0
    for x in range(100):
        total += 1
        moves = [random.randint(NUM_WORLD_STATES)]
        for i in range(NUMBER_OF_PERIODS - 1):
            moves.append(random.choice([0, 1, 2, 3], p=scheduler[moves[i]]))

        # Run simulation
        avg_return, risk, sharpe = simulate_driver(moves, strat)
        
        # Limit to -3 <= sharpe <= 3
        sharpe = max(-3, min(3, sharpe))

        # whether or update R+ or R- 
        if (sharpe < SUCCESS_THRESHOLD):
            win += 1

        print('Current win percentage', win, 'out of', total, ' => ', win/total)




