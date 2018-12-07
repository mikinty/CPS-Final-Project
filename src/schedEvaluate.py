'''
Scheduler Evaluation 

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, NUM_WORLD_STATES, EXPLORATION_CHOICE

from numpy import array, array_equal, zeros, random, divide, nonzero, newaxis
import multiprocessing as mp
from functools import partial
from time import time

from simulation import simulate_driver


# Per thread call to orun a single simulation iteration
def callSimulation(scheduler, strat, i):
    random.seed()

    # generate moves
    moves = [random.randint(NUM_WORLD_STATES)]
    for i in range(NUMBER_OF_PERIODS - 1):
        if random.random() < EXPLORATION_CHOICE:
            moves.append(random.choice(nonzero(scheduler[moves[i]])[0]))
        else:
            moves.append(random.choice([0, 1, 2, 3], p=scheduler[moves[i]]))

    # Run simulation
    avg_return, risk, sharpe = simulate_driver(moves, strat)
    
    # Limit to -3 <= sharpe <= 3
    sharpe = max(-3, min(3, sharpe))

    # whether or update R+ or R-
    if (sharpe < SUCCESS_THRESHOLD):
        index = 1
        sharpe = -sharpe
    else:
        index = 0
        
    # index = 1 if (sharpe < SUCCESS_THRESHOLD) else 0

    # Reinforcement feedback
    R = zeros((2, 4, 4))
    for i in range(len(moves) - 1):
        state, action = moves[i:i+2]

        R[index][state][action] += sharpe

    '''
    row_sums = R[index].sum(axis=1)
    R[index] = divide(R[index], row_sums[:, newaxis], where=row_sums[:, newaxis] != 0)
    R[index] = R[index] * sharpe
    '''

    return R

def schedulerEvaluate(scheduler, strat):
    '''
    Trains the scheduler via RL

    param scheduler: The scheduler we are trying to train

    return: New quality estimates for each state
    '''

    '''
    # parallel version...idk why it doesn't work
    pool = mp.Pool()
    # number of jobs
    jobs = range(SCHEDULER_TRAIN_ITER)

    CS = partial(callSimulation, scheduler, strat)

    R_RES = pool.map(CS, jobs)
    # print(R_RES)
    R = sum(R_RES)

    pool.close()
    '''

    # sequential version
    R = callSimulation(scheduler, strat, 0)
    print(strat)
    for x in range(SCHEDULER_TRAIN_ITER - 1):
        R_RES = callSimulation(scheduler, strat, x)

        # print(R_RES)
        R += R_RES    

    print (R)

    # print(sum(sum(sum(R))))

    # Update scheduler quality estimates
    # Notice that if we didn't encounter a particular state in 
    # our random paths, then scheduler will retain its old value
    #Q = zeros((NUM_WORLD_STATES, NUM_WORLD_STATES))

    '''
    for state, action in R:
        total = (R[0][state][action] + R[1][state][action])

        if total == 0:
            Q[state][action] = 0
        else:
            Q[state][action] = (R[1][state][action] / total)
    '''

    Q = divide(R[1], R[0] + R[1], where=(R[0] + R[1])!=0)
    print('q', Q)

    return Q

