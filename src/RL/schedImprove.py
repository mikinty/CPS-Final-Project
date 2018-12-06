'''
Scheduler Improvement

Michael You
Abhishek Barghava
'''

from CONSTANTS_RL import EXPLORATION_RATE, LEARNING_RATE

# TODO: need to import STATES

def schedulerImprove(scheduler, Q):
    '''
    Greedily chooses the best moves for an improved scheduler

    param scheduler: The scheduler we are improving
    param Q        : Quality factors to use for improvement

    return: New and improved scheduler
    '''
    newScheduler = {}

    for s in STATES:
        bestAction = max(Q[s], key=Q[s].get)

        totalQ = sum(Q[s])

        for a in ACTIONS:
            # the probability we are assigning to the action
            p = EXPLORATION_RATE * (Q[s][a] / totalQ)

            if a == bestAction:
                p += 1 - EXPLORATION_RATE

            newScheduler[(s, a)] = 
                scheduler[(s, a)] * (1 - LEARNING_RATE) + p * LEARNING_RATE

    return newScheduler


