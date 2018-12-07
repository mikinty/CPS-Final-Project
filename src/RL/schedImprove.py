'''
Scheduler Improvement

Michael You
Abhishek Barghava
'''

from .CONSTANTS_RL import EXPLORATION_RATE, LEARNING_RATE, NUM_WORLD_STATES, EPSILON, CONVERGED_Q
import numpy as np

def isConverge(Q):
    return np.array_equal(Q, CONVERGED_Q)

def schedulerImprove(scheduler, Q):
    '''
    Greedily chooses the best moves for an improved scheduler

    param scheduler: The scheduler we are improving
    param Q        : Quality factors to use for improvement

    return: New and improved scheduler
    '''
    if isConverge(Q):
      print('we are converged')
      return scheduler

    newScheduler = np.zeros((NUM_WORLD_STATES, NUM_WORLD_STATES))

    for s in range(NUM_WORLD_STATES):
        bestAction = np.argmax(Q[s])

        totalQ = sum(Q[s])

        # don't update. Shouldn't happen often.
        if totalQ < EPSILON:
            newScheduler[s] = scheduler[s]
            continue

        # print('old', scheduler[s])
        # print('bestAction', bestAction)
        for a in range(NUM_WORLD_STATES):

            # the probability we are assigning to the action
            p = EXPLORATION_RATE * (Q[s][a] / totalQ)

            if a == bestAction:
                p += 1 - EXPLORATION_RATE

            # print(p, end=' ')

            newScheduler[s][a] = scheduler[s][a] * (1 - LEARNING_RATE) + p * LEARNING_RATE

        # print('new', newScheduler[s])
    return newScheduler


