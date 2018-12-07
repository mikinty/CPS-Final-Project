'''
Runs the RL algorithm and simulation code to train the scheduler

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, NUM_WORLD_STATES, NUM_TRAIN_ITER
from schedEvaluate import schedulerEvaluate
from RL.schedImprove import schedulerImprove

import pickle
import sys


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print('Need 2 arguments: STRATEGY SCHEDULER')
        sys.exit() 

    strat = sys.argv[1]
    SCHEDULER_FILE = './schedulers/'+ sys.argv[2] + '.pickle'

    print('Starting Training on', SCHEDULER_FILE)

    # Load scheduler file
    pickle_in = open(SCHEDULER_FILE, 'rb')
    scheduler = pickle.load(pickle_in)
    pickle_in.close()

    print(scheduler)

    # train the scheduler
    for i in range(NUM_TRAIN_ITER):
        print('Iteration', i)
        # Compute new quality factor
        Q = schedulerEvaluate(scheduler, strat)

        # Update scheduler with new quality estimates
        scheduler = schedulerImprove(scheduler, Q)

        pickle_out = open(SCHEDULER_FILE, 'wb')
        
        pickle.dump(scheduler, pickle_out)
        pickle_out.close()

        print(scheduler)

    print('Done Training. Saved new scheduler to', SCHEDULER_FILE)
