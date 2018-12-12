'''
Scheduler SMC

Michael You
Abhishek Barghava
'''

import pickle
import sys

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, NUM_WORLD_STATES, SIM_TRAIN_ITER 

from numpy import array, zeros, random, divide, zeros, arange
import multiprocessing as mp
from functools import partial

from simulation_adv import simulate_driver
from dynamic_strats import long_short_strat

MC_ITER = 100

# Returns True if the scheduler beats strat
def callSimulation(scheduler, strat, i):
    # generate moves
    moves = [random.randint(NUM_WORLD_STATES)]
    for i in range(NUMBER_OF_PERIODS - 1):
        moves.append(random.choice(arange(NUM_WORLD_STATES), p=scheduler[moves[i]]))

    numSim = SIM_TRAIN_ITER

    sharpe = 0
    # Run simulation
    for x in range(numSim):
        avg_return, risk, tsharpe = simulate_driver(moves, strat, long_short_strat)
        sharpe += tsharpe

    sharpe = sharpe / numSim

    print(sharpe < SUCCESS_THRESHOLD, sharpe, i)

    # whether or update R+ or R- 
    return (sharpe < SUCCESS_THRESHOLD)


def runMC(scheduler, strat, numIter):
    # Spawn processes to do task in parallel
    pool = mp.Pool()

    # number of jobs
    jobs = range(numIter)

    CS = partial(callSimulation, scheduler, strat)

    WIN_RES = pool.map(CS, jobs)

    NUM_WINS = sum(WIN_RES)

    pool.close()

    print('Scheduler win percentage', NUM_WINS, 'out of', len(jobs), ' => ', NUM_WINS/len(jobs))
    
    return NUM_WINS/len(jobs)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Need 2 arguments: STRAT SCHEDULER_FILE')
        sys.exit()

    # Parse arguments
    strat = sys.argv[1]
    SCHEDULER_FILE = './schedulers/'+ sys.argv[2] + '.pickle'

    print('Calculating success of', SCHEDULER_FILE)

    # Load scheduler file
    pickle_in = open(SCHEDULER_FILE, 'rb')
    scheduler = pickle.load(pickle_in)
    pickle_in.close()
    
    '''
    # test optimal scheduler (always chooses bad world state)
    cheduler = array([[0, 0, 1,   0], 
                       [0, 0, 0.5, 0.5],
                       [0, 0, 0.5, 0.5 ],
                       [0, 0, 0,   1]])
    '''


    runMC(scheduler, strat, MC_ITER)


