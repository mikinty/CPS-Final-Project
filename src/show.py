import pickle
import sys

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Need 1 argument: SCHEDULER')
        sys.exit()


    SCHEDULER_FILE = './schedulers/' + sys.argv[1] + '.pickle'

    pickle_in = open(SCHEDULER_FILE, 'rb')
    scheduler = pickle.load(pickle_in)
    pickle_in.close()

    print('Current Scheduler')
    print(scheduler)
