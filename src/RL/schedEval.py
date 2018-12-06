# This file contains the scheduler evaluation code
#
# Michael You
# Abhishek Barghava

from CONSTANTS_RL import *

def schedulerTrain(scheduler):
    '''
    Trains the scheduler via RL

    :param scheduler: The scheduler we are trying to train

    :return: None, modifies scheduler with new quality estimates
    '''
    for i in range(SCHEDULER_TRAIN_ITER):
        '''
        Sample minimal sufficient path \pi from \mathcal{M}^\sigma
        
        path:    (state, action) List
                 The path taken during this run. 
        success: Bool
                 whether we ended up in a success state (True or False)
        '''
        # is this a random path?
        path, success = sim()

        # whether or update R+ or R- 
        index = 1 if success else 0

        # Reinforcement feedback
        R = {}
        for state, action in path:
            if (state, action) not in R:
                R[(state, action)] = [0, 0]

            R[(state, action)][index] += 1

    # Update scheduler quality estimates
    for key in R:
        scheduler[key] = (R[key][1] / (R[key][0] + R[key][1]))

