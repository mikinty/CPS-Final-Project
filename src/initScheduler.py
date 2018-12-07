'''
Initializes the scheduler to be uniform

Michael You
Abhishek Barghava
'''
import sys
import pickle
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need 1 argument: SCHEDULER_FILE')
        sys.exit()

    SCHEDULER_FILE = './schedulers/'+ sys.argv[1] + '.pickle'

    print('Creating scheduler matrix', SCHEDULER_FILE)

    # ''' uniform
    scheduler = np.array([[0.5, 0, 0.5, 0],
                          [0.25, 0.25, 0.25, 0.25],
                          [0.25, 0.25, 0.25, 0.25],
                          [0, 0.5, 0, 0.5]])
    '''

    scheduler = np.array([[0.1, 0, 0.9, 0],
                          [0.1, 0.1, 0.4, 0.4],
                          [0.1, 0.1, 0.4, 0.4],
                          [0, 0.1, 0, 0.9]])
    
    '''

    pickle_out = open(SCHEDULER_FILE, 'wb')
    pickle.dump(scheduler, pickle_out)
    pickle_out.close()

