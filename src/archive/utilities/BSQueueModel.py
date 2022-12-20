"""
This script constitutes the evaluation of an $M/G/N_{K}$ Queue Model at the Base Station [Custom Implementation]

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logging level

import time
import warnings
import numpy as np
import tensorflow as tf
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

"""
Global settings
"""
# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Numpy seed
np.random.seed(10191)
# Tensorflow seed
tf.random.set_seed(10191)

"""
CONFIGURATIONS-I: NUMBER OF ORTHOGONAL CHANNELS AT THE BS ($N_{K}$) | NUMBER OF GN REQUESTS ($N_{GN}$)
"""
data_payload_lengths = [0.1e6, 1e6, 5e6]
request_arrival_rates = {0.1e6: 10, 1e6: 1.3333e-2, 5e6: 6.6666e-3}
number_of_channels, number_of_requests, comm_delay_min, comm_delay_max, sleep_seconds = 5, 100, 1.0, 4.0, 0.01

"""
Global resources
"""
queue = tf.Variable(tf.zeros(shape=(number_of_requests,), dtype=tf.int8), dtype=tf.int8)
wait_times = tf.Variable(tf.zeros(shape=(number_of_requests,), dtype=tf.float64), dtype=tf.float64)
lock, server_pool = Lock(), tf.Variable(tf.zeros(shape=(number_of_channels,), dtype=tf.int8), dtype=tf.int8)
service_times = tf.Variable(tf.random.uniform(shape=(number_of_requests,), dtype=tf.float64, minval=comm_delay_min,
                                              maxval=comm_delay_max), dtype=tf.float64)


def service_request(request_number, available_servers):
    """
    Service the GN request: Consume | Occupy a server | Sleep for the service period | Un-occupy the server

    :param request_number: The GN request index for a global post-processing assignment of its service delay
    :param available_servers: A tensor of available servers at this point in the emulation

    :return: The server index (>0 | SERVED) or (-1 | NOT SERVED)
    """
    server = available_servers[0, 0].numpy() if tf.not_equal(tf.size(available_servers), 0) else -1
    # A server is available
    if server != -1:
        tf.compat.v1.assign(queue[request_number], 1, use_locking=True)
        tf.compat.v1.assign(server_pool[server], 1, use_locking=True)
        time.sleep(service_times[request_number].numpy())
        tf.compat.v1.assign(server_pool[server], 0, use_locking=True)
    return server


# The life of a GN communication request
def life_of_a_request(start_time, request_number, call_number):
    """
    A thread-safe, executor-handled routine that emulates the life of a GN request

    :param start_time: The time at which the processing of this GN request started
    :param request_number: The GN request index for a global post-processing assignment of its service delay
    :param call_number: This function call's place in the overall process hierarchy of the GN request under analysis

    :return: A boolean indicating whether the request has been served
    """
    served = False
    while not served:
        with lock:
            available_servers = tf.where(tf.equal(server_pool, 0))
            queue_go_ahead = tf.equal(queue[request_number - 1], 1) if request_number > 0 and call_number == 0 else True
        # FIFO: Ready to Pop!
        if queue_go_ahead:
            # All servers are busy
            if service_request(request_number, available_servers) == -1:
                print(f'Call Number = {call_number} | Waiting {request_number}')
                time.sleep(sleep_seconds)
                served = life_of_a_request(start_time, request_number, call_number + 1)
            # Request served
            else:
                print(f'Call Number = {call_number} | Served {request_number}!')
                tf.compat.v1.assign(wait_times[request_number], (time.time_ns() - start_time) / 1e9, use_locking=True)
                served = True
        # FIFO: Waiting for the request in front of me to Pop!
        else:
            time.sleep(sleep_seconds)
            continue
    return served
    # The request's life ends here...:-(


# The memory-less arrival process of GN communication requests
def simulate_poisson_arrivals(payload_length):
    """
    Simulate the Poisson arrival process of the active communication requests originating from the GNs in the cell

    :param: The data payload length input which serves as a key into the "arrival_rates" dictionary in order to select
            a payload-appropriate arrival rate

    :return: A list of GN active communication request arrival times
    """
    request_number, arrival_rate, arrival_times = 0, request_arrival_rates[payload_length], []
    arrival_time = (-np.log(1 - np.random.random_sample())) / arrival_rate
    while request_number < number_of_requests:
        request_number += 1
        arrival_times.append(arrival_time)
        arrival_time += (-np.log(1 - np.random.random_sample())) / arrival_rate
    return arrival_times


# Run Trigger
if __name__ == '__main__':
    print('[INFO] BSQueueModel main: Starting the evaluation of the M/G/N Queueing System at the Base Station...')
    data_payload_length = data_payload_lengths[0]  # 0.1 Mb data payload example
    request_arrival_times = simulate_poisson_arrivals(data_payload_length)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, request_arrival_time in enumerate(request_arrival_times):
            print(f'Root Call | Arrived {index}!')
            executor.submit(life_of_a_request, time.time_ns(), index, 0)
            time.sleep(request_arrival_times[index + 1] - request_arrival_time
                       if index < number_of_requests - 1 else 0.0)
    print(f'[INFO] BSQueueModel main: The wait delays associated with the {number_of_channels} servers at the '
          f'Base Station for {number_of_requests} GN requests of data payload size {data_payload_length} are '
          f'{wait_times.numpy()}')
    print(f'[INFO] BSQueueModel main: The total service delays associated with the {number_of_channels} servers at the '
          f'Base Station for {number_of_requests} GN requests of data payload size {data_payload_length} are '
          f'{tf.add(service_times, wait_times).numpy()}')
# The evaluations of the ($M/G/N_{K}$) Queueing Model a the Base Station end here.
