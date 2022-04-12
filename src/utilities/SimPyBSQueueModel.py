"""
This script constitutes the evaluation of an $M/G/N_{K}$ Queue Model at the Base Station [SimPy Implementation]

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

# The imports
import numpy as np
from simpy import *

# Numpy random seed
np.random.seed(10191)

# Configurations
number_of_requests, number_of_channels, arrival_rate = 10000, 10, 1.67e-2
wait_times, service_times = [], [147.0 for _ in range(number_of_requests)]


# A GN request
def gn_request(env, num, chs):
    """
    Process the generated GN request

    :param env: The SimPy environment wherein the queueing model is being evaluated
    :param num: The GN request index
    :param chs: The number of orthogonal BS channels (SimPy Resources) in this evaluation
    """
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])
    with chs[k].request() as req:
        yield req
        wait_times.append(env.now - arrival_time)
        yield env.timeout(service_times[num])


# Simulate Poisson arrivals and process them
def arrivals(env, chs):
    """
    Simulate Poisson arrivals and process the requests according to their service times

    :param env: The SimPy environment wherein the queueing model is being evaluated
    :param chs: The number of orthogonal BS channels (SimPy Resources) in this evaluation
    """
    n_r, arr = number_of_requests, arrival_rate
    for num in range(n_r):
        env.process(gn_request(env, num, chs))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


# Run Trigger
environment = Environment()
channels = [Resource(environment) for _ in range(number_of_channels)]
environment.process(arrivals(environment, channels))
environment.run()
print(f'[INFO] SimpyBSQueueModel main: Average Queue Wait Time = {np.mean(wait_times)}')
