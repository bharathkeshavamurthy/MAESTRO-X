"""
This script constitutes functional & performance evaluations of the MAESTRO augmented orchestration policy for a
circular cell of radius $a$ meters consisting of a system of uniformly-distributed Ground Nodes (GNs), a cellular
Base Station (BS) at the center of the cell, and a fleet of Unmanned Aerial Vehicles (UAVs) serving as BS relays.

(preliminary-augmented) MAESTROBeta: Instead of the optimal waiting state policy, the UAVs always return to the center.

For the "fully-optimal" preliminary policy evaluation, see MAESTROAlpha.py.
For the "final-augmented" (JFQ | PHY and DLL prescient scheduler) policy evaluation, see MAESTROeXtended.py.

Data needed for this evaluation has to be in a *.log file with the following information:
    1. MAESTRO evaluation instance UUID identifier;
    2. The size of the data payload being uploaded by the GNs in the cell;
    3. The average power constraint imposed on the UAVs in the fleet during the SMDP optimization process;
    4/5. The waiting states and waiting actions in the SMDP model: UAV positional radius levels & UAV radial velocities;
    6/7. The communication states and communication actions in the SMDP model: UAV initial & requesting GN positions &
       UAV end positions to serve the GN request;
    8. The average communication delays associated with serving GN requests vis-à-vis the optimal SMDP policy (BS/UAV);
    9. The energy consumption costs associated with serving GN requests vis-à-vis the optimal SMDP policy (BS/UAV);
    10. The average communication delays associated with serving GN requests driven by the BS only;
    11. The energy consumption costs associated with serving GN requests driven by the BS only;
    12. The average communication delays associated with serving GN requests driven by the UAV relay;
    13. The energy consumption costs associated with serving GN requests driven by the UAV relay;
    14. The optimal trajectories obtained via HCSO for the UAV serving GN requests, based on their optimal endpoints;
    15. The optimal velocities obtained via HCSO for the UAV serving GN requests, based on their optimal endpoints;
    16. The optimal relay statuses for GN requests: BS service or UAV service determination based on cost minimization;
    17. The optimal waiting state policy determined via SMDP Value Iteration and Projected Sub-gradient Ascent; and
    18. The optimal communication state policy determined via SMDP Value Iteration and Projected Sub-gradient Ascent.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import traceback
import numpy as np
import tensorflow as tf
from simpy import Environment, Resource

"""
CONFIGURATIONS-I: Deployment Model
"""

# The max allowed UAV forward velocity in our model (in m/s)
max_uav_velocity = 55.0

# The number of channels/servers ($N_{K}$) at the BS to handle incoming GN requests
number_of_bs_channels = 10

# The arrival rates of different payloads from the GNs in the cell under analysis ($\Lambda$) in requests/second
arrival_rates = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.5555e-4}

# The UUID identifier of the SMDPEvaluation agent/instance whose functionality and performance is being evaluated here
agent_identifier = "1221d753-0bd1-477f-96c6-10899725037b"

# The initial deployment positions of the UAVs in the fleet | These positional args should be a part of the state space
uav_coords = [tf.Variable([[0.0, 500.0]], dtype=tf.float64),  # UAV-0
              tf.Variable([[-400.0, -300.0]], dtype=tf.float64),  # UAV-1
              tf.Variable([[400.0, -300.0]], dtype=tf.float64)]  # UAV-2

# The number of UAVs ($N_{U}$) in the fleet being optimized | Extract it from the positional arguments given to the UAVs
number_of_uavs = len(uav_coords)

# The weight or multiplication factor for the delay cost associated with serving a GN request from a node (BS/UAV-relay)
omicron = 0.7

"""
SimPy Queueing Routines
"""


def wait(x_u, z_u):
    """
    Waiting state operations of a UAV in the fleet

    :param x_u: The current (x, y) position of the UAV under waiting-state analysis
    :param z_u: The time at which the last served request left the queue of the UAV under waiting-state analysis

    :return: The next/new (x, y) location of the UAV under waiting-state analysis, according to the optimal wait policy
    """
    r_u = tf.norm(x_u, axis=1)
    t_u = time.monotonic() - z_u
    d_u = tf.multiply(max_uav_velocity, t_u)
    # In this augmentation, consider only the "lower" radii-levels (r^)
    s_comm_x = tf.gather(s_comm[:, 0, :], tf.where(tf.less_equal(tf.norm(s_comm[:, 0, :], axis=1), r_u)))
    return s_comm_x[tf.argmin(tf.subtract(d_u, tf.norm(tf.subtract(x_u, s_comm_x), axis=1)))]  # Return closest in r^


# noinspection PyTypeChecker
def gn_request(j, z):
    """
    Generator: Process the GN request

    j: The GN request index for referencing into the various collections
    z: A UAV-indexed dictionary to store the time at which its last known request was serviced
    """
    t, x_g = env.now, s_comm[i_s[j], 1]
    [tf.compat.v1.assign(x_u, wait(x_u, z[u]), validate_shape=True,
                         use_locking=True) if z[u] > 0.0 else None for u, x_u in enumerate(uav_coords)]
    st = [tf.argmin(tf.reduce_sum(tf.norm(s_comm - tf.concat(x_u, x_g, axis=1), axis=1), axis=1)) for x_u in uav_coords]
    tmps = [(u, u_star[s], xi_star[s], delta_s[s], energy_s[s]) for u, s in enumerate(st)]  # UAV-specific tmp variables
    is_uav = sum([m[2] for m in tmps])  # =0 => The BS handles the GN request | =1 => a UAV-relay handles the GN request
    best_uav = min(tmps, key=lambda m: omicron * m[3] + (1 - omicron) * m[4])  # Conflict Resolution with d-e factorings
    k_bs = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in res[:number_of_bs_channels]])  # JSQ-Concept
    k, mu, nrg = (k_bs, min([m[3] for m in tmps]), 0.0) if not is_uav else (number_of_bs_channels + best_uav[0],
                                                                            best_uav[3], best_uav[4])  # BS / UAV-relay?

    # Serve the request | Wait in the queue, if needed
    with res[k].request() as req:
        yield req
        wait_times.append(env.now - t)
        yield env.timeout(mu)

    # Final assignments after this GN request is served
    final_serv_times.append(mu)
    final_energy_costs.append(nrg)
    if is_uav:
        z[k - number_of_bs_channels] = time.monotonic()
        tf.compat.v1.assign(uav_coords[best_uav[0]], best_uav[1], validate_shape=True, use_locking=True)


def arrivals():
    """
    Generator: Simulate Poisson arrivals and process the requests according to their service times
    """
    z = {u: 0.0 for u in range(number_of_uavs)}
    for i in range(n_comm):
        env.process(gn_request(i, z))
        yield env.timeout(-np.log(np.random.random_sample()) / lambda_arr)


"""
SETUP: Log file parse | Argument assignment | Temporary collections | Output initializations
"""

args, filename = list(), "".join(["../perf-logs/", agent_identifier, ".log"])
try:
    with open(filename, 'r') as file:
        args.append(tf.convert_to_tensor(file.readline().strip(), dtype=tf.string))
        for line in file.readlines():
            args.append(tf.convert_to_tensor(line.strip(), dtype=tf.float64))
except Exception as e:
    print(f'[ERROR] MAESTROBeta main: Exception caught while parsing {filename} - '
          f'{traceback.print_tb(e.__traceback__)}')

uid, ell, p_avg, s_wait, a_wait, s_comm, a_comm, delta_s, energy_s = args[:9]
bs_delta_s, bs_energy_s, uav_delta_s, uav_energy_s, p_star, v_star, xi_star, o_star, u_star = args[9:]
env, lambda_arr, n_wait, n_comm = Environment(), arrival_rates[ell.numpy()], s_wait.shape[0], s_comm.shape[0]
res = [Resource(env) for _ in range(number_of_bs_channels + number_of_uavs)]  # Treat as (N_U + N_K) total servers

i_s = tf.random.shuffle([_ for _ in range(n_comm)])
a_wait_opts, a_comm_opts = o_star.numpy(), u_star.numpy()
relays, serv_times, energy_costs = xi_star.numpy(), delta_s.numpy(), energy_s.numpy()
wait_times, final_serv_times, final_energy_costs = list(), list(), list()  # Outputs after conflict resolution (BS/UAVs)

"""
CORE SIMULATION
"""

env.process(arrivals())
env.run()

"""
OUTPUT LOGGING
"""

print(f'[INFO] MAESTROBeta main: MAESTRO eval ID = {uid.numpy().decode()} | Number of UAVs = {number_of_uavs} | '
      f'Number of orthogonal BS channels = {number_of_bs_channels} | Payload size = {ell.numpy() / 1e6} Mb | '
      f'Arrival rate = {lambda_arr} req/s | Average power constraint = {p_avg.numpy() / 1e3} kW | '
      f'Number of wait states = {n_wait} | Number of comm states = {n_comm} | '
      f'1 UAV-relay average service time = {np.mean(serv_times)} s | '
      f'1 UAV-relay average energy consumption = {np.mean(energy_costs) / 1e3} kJ | '
      f'{number_of_uavs} UAV-relay(s) average queue wait time = {np.mean(wait_times)} s | '
      f'{number_of_uavs} UAV-relay(s) average service time = {np.mean(final_serv_times)} s | '
      f'{number_of_uavs} UAV-relay(s) average energy consumption = {np.mean(final_energy_costs) / 1e3} kJ')
# The MAESTRO-Beta policy evaluation for multiple UAV-relays ends here...
