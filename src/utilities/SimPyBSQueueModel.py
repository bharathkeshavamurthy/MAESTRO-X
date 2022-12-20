"""
This script constitutes the evaluation of an M/G/x Queue Model at the Base Station [SimPy Implementation].

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import numpy as np
from simpy import *

"""
Configurations: Simulation parameters
"""
np.random.seed(6)
number_of_requests, number_of_queues, arrival_rate = 1000, 4, 5 / 60
wait_times, service_times = [], [156.7514 for _ in range(number_of_requests)]

"""
Core operations
"""


def gn_request(env, num, chs):
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        wait_times.append(env.now - arrival_time)
        yield env.timeout(service_times[num])


def arrivals(env, chs):
    n_r, arr = number_of_requests, arrival_rate

    for num in range(n_r):
        env.process(gn_request(env, num, chs))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


environment = Environment()
channels = [Resource(environment) for _ in range(number_of_queues)]

environment.process(arrivals(environment, channels))

environment.run()

print(f'[INFO] SimpyBSQueueModel main: Average Queue Wait Time = {np.mean(wait_times)}.')
