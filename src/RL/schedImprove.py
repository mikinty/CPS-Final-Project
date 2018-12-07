'''
Scheduler Improvement

Michael You
Abhishek Barghava
'''

from .CONSTANTS_RL import EXPLORATION_RATE, LEARNING_RATE, NUM_WORLD_STATES
import numpy as np

def schedulerImprove(scheduler, Q):
    '''
    Greedily chooses the best moves for an improved scheduler

    param scheduler: The scheduler we are improving
    param Q        : Quality factors to use for improvement

    return: New and improved scheduler
    '''
    newScheduler = np.zeros((NUM_WORLD_STATES, NUM_WORLD_STATES))

    for s in range(NUM_WORLD_STATES):
        bestAction = np.argmax(Q[s])

        totalQ = sum(Q[s])

        for a in range(NUM_WORLD_STATES):
            # the probability we are assigning to the action
            p = EXPLORATION_RATE * (Q[s][a] / totalQ)

            if a == bestAction:
                p += 1 - EXPLORATION_RATE

            newScheduler[s][a] = scheduler[s][a] * (1 - LEARNING_RATE) + p * LEARNING_RATE

    return newScheduler


