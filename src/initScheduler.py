'''
Initializes the scheduler to be uniform

Michael You
Abhishek Barghava
'''

import pickle
import numpy as np

SCHEDULER_FILE = './schedulers/scheduler1.pickle'

if __name__ == '__main__':
    print('Calculating success of', SCHEDULER_FILE)

    scheduler = np.array([[0.5, 0, 0.5, 0],
                          [0.25, 0.25, 0.25, 0.25],
                          [0.25, 0.25, 0.25, 0.25],
                          [0, 0.5, 0, 0.5]])
    

    pickle_out = open(SCHEDULER_FILE, 'wb')
    pickle.dump(scheduler, pickle_out)
    pickle_out.close()

