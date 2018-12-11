'''
Scheduler Evaluation 

Michael You
Abhishek Barghava
'''

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD, NUM_WORLD_STATES
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, EXPLORATION_CHOICE, SIM_TRAIN_ITER 

from numpy import array, array_equal, zeros, random, divide, nonzero, newaxis, arange
import multiprocessing as mp
from functools import partial
from time import time

from simulation_adv import simulate_driver
from dynamic_strats import mvo_optimal_strat, buy_and_hold_strat, long_short_strat
from smc import runMC
import pickle

# strategy to use
STRAT = buy_and_hold_strat

# Per thread call to orun a single simulation iteration
def callSimulation(scheduler, strat, i):
    random.seed()

    # generate moves
    moves = [random.randint(NUM_WORLD_STATES)]

    for i in range(NUMBER_OF_PERIODS - 1):
        if random.random() < EXPLORATION_CHOICE:
            moves.append(random.choice(nonzero(scheduler[moves[i]])[0]))
        else:
            moves.append(random.choice(arange(NUM_WORLD_STATES), p=scheduler[moves[i]]))

    sharpe = 0
    # Run simulation
    for x in range(SIM_TRAIN_ITER):
        avg_return, risk, tempSharpe = simulate_driver(moves, strat, STRAT)
        sharpe += tempSharpe
    
    sharpe = sharpe / SIM_TRAIN_ITER

    # Limit to -3 <= sharpe <= 3
    sharpe = max(-3, min(3, sharpe)) - SUCCESS_THRESHOLD
    # print(sharpe)
    # whether or update R+ or R-
    if (sharpe < 0):
        index = 1
        sharpe = -sharpe
        # print(moves)
    else:
        index = 0

    # Reinforcement feedback
    R = zeros((2, NUM_WORLD_STATES, NUM_WORLD_STATES))
    for i in range(len(moves) - 1):
        state, action = moves[i:i+2]

        R[index][state][action] += sharpe

    return R

def schedulerEvaluate(scheduler, strat):
    '''
    Trains the scheduler via RL

    param scheduler: The scheduler we are trying to train

    return: New quality estimates for each state
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
    '''

    print (R)

    # print(sum(sum(sum(R))))

    # Update scheduler quality estimates
    # Notice that if we didn't encounter a particular state in 
    # our random paths, then scheduler will retain its old value
    Q = divide(R[1], R[0] + R[1], where=(R[0] + R[1])!=0)
    print('q', Q)

    '''
    # How good is our scheduler?
    prob = runMC(scheduler, strat, 100)

    stat = pickle.load(open('stats/output.pickle', 'rb'))

    stat.append((scheduler, prob))

    pickle.dump(stat, open('stats/output.pickle', 'wb'))

    '''

    return Q

