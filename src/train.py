'''
Runs the RL algorithm and simulation code to train the scheduler

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, NUM_WORLD_STATES, NUM_TRAIN_ITER
from schedEvaluate import schedulerEvaluate
from RL.schedImprove import schedulerImprove

import pickle

SCHEDULER_FILE = './schedulers/scheduler1.pickle'

if __name__ == '__main__':
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
        Q = schedulerEvaluate(scheduler)

        print('improve')
        # Update scheduler with new quality estimates
        scheduler = schedulerImprove(scheduler, Q)

    pickle_out = open(SCHEDULER_FILE, 'wb')
    pickle.dump(scheduler, pickle_out)
    pickle_out.close()

    print('Done Training. Saved new scheduler to', SCHEDULER_FILE)
    print(scheduler)