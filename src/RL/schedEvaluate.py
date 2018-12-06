'''
Scheduler Evaluation 

Michael You
Abhishek Barghava
'''

from CONSTANTS_RL import SCHEDULER_TRAIN_ITER

def schedulerEvaluate(scheduler):
    '''
    Trains the scheduler via RL

    param scheduler: The scheduler we are trying to train

    return: New quality estimates for each state
    '''
    for i in range(SCHEDULER_TRAIN_ITER):
        '''
        Sample minimal sufficient path \pi from \mathcal{M}^\sigma
        
        path:    (state, action) List
                 The path taken during this run. 
        success: Bool
                 whether we ended up in a success state (True or False)
        '''
        # TODO: need code to simulate the MDP path
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
    # Notice that if we didn't encounter a particular state in 
    # our random paths, then scheduler will retain its old value
    Q = {}

    for state, action in R:
        total = (R[(state, action)][0] + R[(state, action)][1])

        if total == 0:
            Q[state][action] = 0
        else:
            Q[state][action] = (R[(state, action)][1] / total)

    return Q