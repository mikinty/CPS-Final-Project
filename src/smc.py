'''
Scheduler SMC

Michael You
Abhishek Barghava
'''

import pickle
import sys

from CONSTANTS_MAIN import NUMBER_OF_PERIODS, SUCCESS_THRESHOLD
from RL.CONSTANTS_RL import SCHEDULER_TRAIN_ITER, NUM_WORLD_STATES

from numpy import array, zeros, random, divide, zeros, arange
import multiprocessing as mp
from functools import partial

from simulation import simulate_driver

MC_ITER = 1000

# Returns True if the scheduler beats strat
def callSimulation(scheduler, strat, i):
    # generate moves
    moves = [random.randint(NUM_WORLD_STATES)]
    for i in range(NUMBER_OF_PERIODS - 1):
        moves.append(random.choice(arange(NUM_WORLD_STATES), p=scheduler[moves[i]]))

    # Run simulation
    avg_return, risk, sharpe = simulate_driver(moves, strat)

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


