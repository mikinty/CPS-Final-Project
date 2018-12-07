'''
Scheduler Evaluation 

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, NUM_WORLD_STATES
from numpy import array, zeros

from world_states import sim

def schedulerEvaluate(scheduler):
    '''
    Trains the scheduler via RL

    param scheduler: The scheduler we are trying to train

    return: New quality estimates for each state
    '''
    for i in range(SCHEDULER_TRAIN_ITER):
        # generate moves
        moves = [random.randint(NUM_WORLD_STATES)]
        for i in range(NUMBER_OF_PERIODS - 1):
            moves.append(random.choice([0, 1, 2, 3], p=scheduler[moves[i]]))

        # Run simulation
        success = sim(moves)
        
        # whether or update R+ or R- 
        index = 1 if (success > SUCCESS_THRESHOLD) else 0

        # Reinforcement feedback
        R = {}
        for i in range(len(moves) - 1):
            state, action = moves[i:i+2]
            
            if (state, action) not in R:
                R[(state, action)] = [0, 0]

            R[(state, action)][index] += 1

    # Update scheduler quality estimates
    # Notice that if we didn't encounter a particular state in 
    # our random paths, then scheduler will retain its old value
    Q = zeros((NUM_WORLD_STATES, NUM_WORLD_STATES))

    for state, action in R:
        total = (R[(state, action)][0] + R[(state, action)][1])

        if total == 0:
            Q[state][action] = 0
        else:
            Q[state][action] = (R[(state, action)][1] / total)

    return Q