"""
This script evaluates the performance of a DDQN implementation of the Multi-UAV Data Harvesting Problem.

Please see uav-data-harvesting git submodule for the code corresponding to this state-of-the-art framework. Also, check
out the results section in the parent repository for screenshots and logs corresponding to the execution of this
framework on our ASU ECEE NVIDIA 4xA100 GPU cluster.

Reference Paper:

    @article{DDQN,
            author = {Harald Bayerlein and Mirco Theile and Marco Caccamo and David Gesbert},
            title = {Multi-{UAV} Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning},
            journal = {IEEE Open Journal of the Communications Society},
            year = {2021},
            volume = {2},
            pages = {1171-1187},
            doi = {10.1109/OJCOMS.2021.3081996}
    }

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import os

"""
Configurations-I: Tensorflow logging
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import warnings
import numpy as np
import tensorflow as tf
from collections import namedtuple
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# Numpy seed
np.random.seed(6)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Deployment settings | Extracted from policy on ASU ECEE EXXACT GPU cluster
"""
ip_dir = '../../../logs/evaluations/'
data_payload_sizes, number_of_workers = [1e6, 10e6, 100e6], 1024
arrival_rates_r, n_uavs, uav_height, bs_height = {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}, 3, 200.0, 80.0

'''
TODO: Read these directly from the policy file as tensors instead of hard-coding them from the cluster node
'''

''' GNs deployment | We treat the BS as GN #-1 to emulate the forward phase of the D&F protocol '''

gn_positions = {}
for gn_idx in range(-1, 10, 1):
    ip_file = f'{ip_dir}{gn_idx}.log'

    with open(ip_file, 'r') as file:
        # noinspection RegExpUnnecessaryNonCapturingGroup
        gn_positions[gn_idx] = tf.strings.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)',
                                                               file.readline().strip()), tf.float64)

''' UAV(s) schedules '''

uav_schedules = {}
for uav_idx in range(n_uavs):
    ip_file = f'{ip_dir}{uav_idx}.log'

    with open(ip_file, 'r') as file:
        # noinspection RegExpUnnecessaryNonCapturingGroup
        uav_schedules[uav_idx] = tf.strings.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)',
                                                                 file.readline().strip()), tf.int8)

''' Corresponding UAV(s) trajectories '''

uav_positions = {}
for uav_idx in range(n_uavs):
    uav_positions[uav_idx] = {}
    for gn_idx in uav_schedules[uav_idx]:
        ip_file = f'{ip_dir}{uav_idx}_{gn_idx}.log'

        with open(ip_file, 'r') as file:
            # noinspection RegExpUnnecessaryNonCapturingGroup
            uav_positions[uav_idx][gn_idx] = tf.strings.to_number(re.findall(
                r'-?\d*\.?\d+e[+-]?\d+|[-+]?(?:\d*\.*\d+)', file.readline().strip().replace('\\n', '')), tf.float64)

"""
Configurations-III: Traffic generation model
"""
depl_env, rf, le_l, le_m, le_h = 'rural', n_uavs, 1, 10, 100
arrival_rates_l = {_k: _v * rf * le_l for _k, _v in arrival_rates_r.items()}
arrival_rates_m = {_k: _v * rf * le_m for _k, _v in arrival_rates_r.items()}
arrival_rates_h = {_k: _v * rf * le_h for _k, _v in arrival_rates_r.items()}

"""
Configurations-IV: Channel model
"""

'''
TODO: Change k1, k2, z1, and z2 according to the deployment environment
TODO: Change bw and n_c according to the deployment environment (Verizon LTE/LTE-A/5G)
'''
if depl_env == 'rural':
    bw, n_c, k1, k2, z1, z2, arrival_rates = 10e6, 2, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_l
elif depl_env == 'suburban':
    bw, n_c, k1, k2, z1, z2, arrival_rates = 20e6, 4, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_m
else:
    bw, n_c, k1, k2, z1, z2, arrival_rates = 40e6, 8, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_h

bw_, num_req = bw / n_c, int(1e4)
s_0, al, anl, kp, ra_conf, ra_tol = linear((5e6 * 40) / bw_), 2.0, 2.8, 0.2, 10, 1e-10

"""
Configurations-IV: UAV mobility power consumption model
"""
vel, u_tip, v_0, p_1, p_2, p_3 = 10.0, 200.0, 7.2, 580.65, 790.6715, 0.0073

"""
Utilities
"""


def evaluate_power_consumption():
    """
    UAV mobility power consumption
    """
    return (p_1 * (1 + ((3 * (vel ** 2)) / (u_tip ** 2)))) + (p_3 * (vel ** 3)) + \
        (p_2 * (((1 + ((vel ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((vel ** 2) / (2 * (v_0 ** 2)))) ** 0.5)


def gn_request(env, num, chs, w_times, serv_times):
    """
    Simpy queueing model: GN request
    """
    arr_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        w_times.append(env.now - arr_time)
        yield env.timeout(serv_times[num])


def arrivals(env, chs, n_r, arr, w_times, serv_times):
    """
    Simpy queueing model: Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request(env, num, chs, w_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


class LinkPerformance(object):
    """
    Link performance
    """

    def __init__(self, channel_bandwidth, reference_snr_at_1_meter,
                 los_path_loss_exponent, nlos_path_loss_exponent, nlos_attenuation_constant,
                 los_rician_factor_1, los_rician_factor_2, propagation_environment_parameter_1,
                 propagation_environment_parameter_2, convergence_confidence, bisection_method_tolerance):
        self.channel_bandwidth = channel_bandwidth
        self.los_rician_factor_1 = los_rician_factor_1
        self.los_rician_factor_2 = los_rician_factor_2
        self.convergence_confidence = convergence_confidence
        self.los_path_loss_exponent = los_path_loss_exponent
        self.nlos_path_loss_exponent = nlos_path_loss_exponent
        self.nlos_attenuation_constant = nlos_attenuation_constant
        self.reference_snr_at_1_meter = reference_snr_at_1_meter
        self.bisection_method_tolerance = bisection_method_tolerance
        self.propagation_environment_parameter_1 = propagation_environment_parameter_1
        self.propagation_environment_parameter_2 = propagation_environment_parameter_2

        self.evaluation_output = namedtuple('link_performance_evaluation_output',
                                            ['los_throughputs', 'nlos_throughputs',
                                             'average_throughputs', 'average_delays', 'aggregated_average_delay'])

    def __enter__(self):
        return self

    # noinspection PyMethodMayBeStatic
    def __f_z(self, z):
        b = self.channel_bandwidth
        return b * np.log2(1 + (0.5 * (z ** 2)))

    # noinspection PyMethodMayBeStatic
    def __marcum_q(self, df, nc, x):
        return 1 - ncx2.cdf(x, df, nc)

    def __f(self, z, *args):
        df, nc, y = args
        f_z = self.__f_z(z)
        q_m = self.__marcum_q(df, nc, (y * (z ** 2)))
        ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
        ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf

        return -ln_f_z - ln_q_m

    # noinspection PyMethodMayBeStatic
    def __bisect(self, f, df, nc, y, low, high, tolerance):
        args = (df, nc, y)
        mid, converged, conf, conf_th = 0.0, False, 0, self.convergence_confidence

        while not converged or conf < conf_th:
            mid = (high + low) / 2
            if (f(low, *args) * f(high, *args)) > 0.0:
                low = mid
            else:
                high = mid
            converged = abs(high - low) < tolerance
            conf += 1 if converged else -conf

        return mid

    def __z(self, gamma):
        b = self.channel_bandwidth
        return np.sqrt(2 * ((2 ** (gamma / b)) - 1))

    def __u(self, gamma, d, los):
        b, gamma_ = self.channel_bandwidth, self.reference_snr_at_1_meter
        alpha, alpha_, kappa = self.los_path_loss_exponent, self.nlos_path_loss_exponent, self.nlos_attenuation_constant

        return ((2 ** (gamma / b)) - 1) / (gamma_ * 1 if los else kappa * (d ** -alpha if los else -alpha_))

    def __evaluate_los_throughput(self, d, phi, r_los):
        k_1, k_2 = self.los_rician_factor_1, self.los_rician_factor_2
        k, alpha = k_1 * np.exp(k_2 * phi), self.los_path_loss_exponent
        b, gamma_ = self.channel_bandwidth, self.reference_snr_at_1_meter
        df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), self.bisection_method_tolerance

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ * (d ** -alpha))), t)

        tf.compat.v1.assign(r_los, self.__f_z(z_star), use_locking=True)

    def __evaluate_nlos_throughput(self, d, r_nlos):
        alpha_, t = self.nlos_path_loss_exponent, self.bisection_method_tolerance
        b, gamma_, kappa = self.channel_bandwidth, self.reference_snr_at_1_meter, self.nlos_attenuation_constant

        df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(b * np.log2(1 + (
                                       rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa * (d ** -alpha_))), t)

        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), use_locking=True)

    def __calculate_adapted_throughput(self, d, phi, r_los, r_nlos, num_workers):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.submit(self.__evaluate_nlos_throughput, d, r_nlos)
            executor.submit(self.__evaluate_los_throughput, d, phi, r_los)

    def __average_throughputs(self, d_s, phi_s, num_workers):
        r_los_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)
        z_1, z_2 = self.propagation_environment_parameter_1, self.propagation_environment_parameter_1

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, (d, phi) in enumerate(zip(d_s, phi_s)):
                executor.submit(self.__calculate_adapted_throughput, d, phi, r_los_s[i], r_nlos_s[i], num_workers)

        phi_degrees = (180.0 / np.pi) * phi_s
        p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
        p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)

        return r_los_s, r_nlos_s, tf.add(tf.multiply(p_los, r_los_s), tf.multiply(p_nlos, r_nlos_s))

    # noinspection PyMethodMayBeStatic
    def __average_delays(self, gn_id, num_gns, p_lens, r_bars):
        delta_bars_p = {
            p_len: tf.divide(tf.constant(p_len * num_gns if gn_id == -1 else p_len,
                                         shape=r_bars.shape, dtype=tf.float64), r_bars) for p_len in p_lens}

        delta_bars_agg = {
            p_len: tf.reduce_sum(delta_bars).numpy()
            for p_len, delta_bars in delta_bars_p.items()}

        return delta_bars_p, delta_bars_agg

    def evaluate(self, gn_id, num_gns, d_s, phi_s, p_lens, num_workers):
        r_los_s, r_nlos_s, r_bars = self.__average_throughputs(d_s, phi_s, num_workers)

        delta_bars_p, delta_bars_agg = self.__average_delays(gn_id, num_gns, p_lens, r_bars)

        return self.evaluation_output(los_throughputs=r_los_s,
                                      nlos_throughputs=r_nlos_s,
                                      average_throughputs=r_bars,
                                      average_delays=delta_bars_p, aggregated_average_delay=delta_bars_agg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] LinkPerformance Termination: Tearing things down - '
              f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {exc_tb}.')


"""
Core operations
"""


def multiple_uav_relays(payload_sizes, gn_coords):
    gu_delays = {k: {_k: {} for _k in v.keys()} for k, v in uav_positions.items()}

    gu_xy_distances = {k: {_k: tf.norm(tf.subtract(gn_coords[_k], _v), axis=1)
                           for _k, _v in v.items()} for k, v in uav_positions.items()}

    gu_heights = {k: {_k: tf.constant(uav_height, shape=_v.shape, dtype=tf.float64)
                      for _k, _v in v.items()} for k, v in gu_xy_distances.items()}

    gu_distances = {k: {_k: tf.sqrt(tf.add(tf.square(gu_xy_distances[k][_k]), tf.square(gu_heights[k][_k])))
                        for _k in v.keys()} for k, v in gu_xy_distances.items()}

    gu_angles = {k: {_k: tf.asin(tf.divide(gu_heights[k][_k], gu_distances[k][_k]))
                     for _k in v.keys()} for k, v in gu_xy_distances.items()}

    lperf = LinkPerformance(bw_, s_0, al, anl, kp, k1, k2, z1, z2, ra_conf, ra_tol)

    for k, v in gu_delays.items():
        for _k, a_v in v.items():
            dists, angles = gu_distances[k][_k], gu_angles[k][_k]
            gu_delays[k][_k] = lperf.evaluate(_k, len(v.keys()) - 1, dists, angles,
                                              payload_sizes, number_of_workers).aggregated_average_delay

    return gu_delays


def evaluate():
    delays = multiple_uav_relays(data_payload_sizes, gn_positions)

    delays_mod = {p: {k: np.random.permutation(
        np.repeat([v[_k][p] for _k in v.keys()], num_req)) for k, v in delays.items()} for p in data_payload_sizes}

    for p, v in delays_mod.items():
        p_size = p / 1e6
        totals = {_u: 0.0 for _u in range(n_uavs)}
        services, waits = {_u: 0.0 for _u in range(n_uavs)}, {_u: 0.0 for _u in range(n_uavs)}

        for _k, _v in v.items():
            _waits, e = [], Environment()

            e.process(arrivals(e, [Resource(e) for _ in range(n_c)], len(_v), arrival_rates[p], _waits, _v))

            e.run()

            services[_k] = np.mean(_v)
            waits[_k] = np.mean(_waits)
            totals[_k] = services[_k] + waits[_k]

        print(f'[DEBUG] DDQNEvaluation evaluate: Payload Size = {p_size} Mb | '
              f'Average Wait Time = {np.mean([_ for _ in waits.values()])} seconds.')

        print(f'[DEBUG] DDQNEvaluation evaluate: Payload Size = {p_size} Mb | '
              f'Average Communication Service Time = {np.mean([_ for _ in services.values()])} seconds.')

        print('[INFO] DDQNEvaluation evaluate: '
              f'[{n_uavs}] UAV Relays | Payload Size = {p_size} Mb | '
              f'UAV Power Consumption = {evaluate_power_consumption() / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean([_ for _ in totals.values()])} seconds.\n')


# Run Trigger
if __name__ == '__main__':
    evaluate()
