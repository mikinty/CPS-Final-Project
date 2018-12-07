'''
Runs the RL algorithm and simulation code to train the scheduler

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, NUM_WORLD_STATES, NUM_TRAIN_ITER
from schedEvaluate import schedulerEvaluate
from RL.schedImprove import schedulerImprove

from numpy import random
import pickle

SCHEDULER_FILE = './schedulers/scheduler1.pickle'

if __name__ == '__main__':
    print('Starting Simulation')

    # Load scheduler file
    pickle_in = open(SCHEDULER_FILE, 'rb')
    scheduler = pickle.load(pickle_in)
    pickle_in.close()

    # train the scheduler
    for i in range(NUM_TRAIN_ITER):
        # Compute new quality factor
        Q = schedulerEvaluate(scheduler, moves, result)

        # Update scheduler with new quality estimates
        scheduler = schedulerImprove(scheduler, Q)

    

