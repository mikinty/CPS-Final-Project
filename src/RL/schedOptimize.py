'''
Scheduler Optimization

Michael You
Abhishek Barghava
'''

from schedEvaluate import schedulerEvaluate
from schedImprove import schedulerImprove

def schedulerOptimize(scheduler, iterations):
    for i in range(iterations):
        # TODO: MC induced by MDP M and scheduler \sigma ???

        # Compute new quality factor
        Q = schedulerEvaluate(scheduler)

        # Update scheduler with new quality estimates
        scheduler = schedulerImprove(scheduler, Q)

