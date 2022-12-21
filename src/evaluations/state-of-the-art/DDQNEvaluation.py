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

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Deployment parameters | Extracted from policy on ASU ECEE EXXACT GPU cluster
"""

np.random.seed(6)
scaling_factor, data_payload_sizes, number_of_workers = 10, [1e6, 10e6, 100e6], 1024
arrival_rates_r, n_uavs, uav_height, bs_height = {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}, 3, 200.0, 80.0

'''
TODO: Read these directly from the policy file as tensors instead of hard-coding them from the cluster node
'''

gn_heights = {-1: tf.constant(bs_height, dtype=tf.float64),
              0: tf.constant(0.0, dtype=tf.float64), 1: tf.constant(0.0, dtype=tf.float64),
              2: tf.constant(0.0, dtype=tf.float64), 3: tf.constant(0.0, dtype=tf.float64),
              4: tf.constant(0.0, dtype=tf.float64), 5: tf.constant(0.0, dtype=tf.float64),
              6: tf.constant(0.0, dtype=tf.float64), 7: tf.constant(0.0, dtype=tf.float64),
              8: tf.constant(0.0, dtype=tf.float64), 9: tf.constant(0.0, dtype=tf.float64)}

gn_positions = {-1: tf.constant([0.0, 0.0], dtype=tf.float64),
                0: tf.constant([0.0, 9.0], dtype=tf.float64), 1: tf.constant([4.0, 14.0], dtype=tf.float64),
                2: tf.constant([9.0, 15.0], dtype=tf.float64), 3: tf.constant([14.0, 2.0], dtype=tf.float64),
                4: tf.constant([14.0, 50.0], dtype=tf.float64), 5: tf.constant([20.0, 43.0], dtype=tf.float64),
                6: tf.constant([42.0, 27.0], dtype=tf.float64), 7: tf.constant([42.0, 24.0], dtype=tf.float64),
                8: tf.constant([47.0, 8.0], dtype=tf.float64), 9: tf.constant([48.0, 22.0], dtype=tf.float64)}

uav_0_trajectory = {2: tf.constant([[24.0, 24.0], [24.0, 23.0], [23.0, 23.0], [23.0, 22.0], [22.0, 22.0], [21.0, 22.0],
                                    [20.0, 22.0], [19.0, 22.0], [19.0, 21.0], [18.0, 21.0], [17.0, 21.0], [17.0, 20.0],
                                    [16.0, 20.0], [15.0, 20.0], [15.0, 19.0], [14.0, 19.0], [14.0, 18.0], [13.0, 18.0],
                                    [13.0, 17.0], [12.0, 17.0], [11.0, 17.0], [10.0, 17.0]], dtype=tf.float64),
                    1: tf.constant([[9.0, 17.0], [9.0, 16.0], [8.0, 16.0], [7.0, 16.0], [6.0, 16.0], [5.0, 16.0],
                                    [4.0, 16.0], [4.0, 15.0]], dtype=tf.float64),
                    0: tf.constant([[3.0, 15.0], [3.0, 14.0], [3.0, 13.0], [2.0, 13.0], [2.0, 12.0], [2.0, 11.0],
                                    [2.0, 10.0], [2.0, 11.0], [2.0, 10.0]], dtype=tf.float64),
                    3: tf.constant([[2.0, 9.0], [2.0, 8.0], [2.0, 7.0], [2.0, 6.0], [3.0, 6.0], [4.0, 6.0],
                                    [5.0, 6.0], [5.0, 5.0], [6.0, 5.0], [7.0, 5.0], [8.0, 5.0], [9.0, 5.0],
                                    [10.0, 5.0], [11.0, 5.0], [12.0, 5.0], [13.0, 5.0], [14.0, 5.0],
                                    [15.0, 5.0]], dtype=tf.float64),
                    -1: tf.constant([[14.0, 5.0], [13.0, 5.0], [12.0, 5.0], [11.0, 5.0], [10.0, 5.0], [9.0, 5.0],
                                     [8.0, 5.0], [7.0, 5.0], [6.0, 5.0], [5.0, 5.0], [4.0, 5.0], [3.0, 5.0],
                                     [2.0, 5.0], [2.0, 4.0], [2.0, 3.0]], dtype=tf.float64)}

uav_1_trajectory = {5: tf.constant([[24.0, 26.0], [24.0, 27.0], [24.0, 28.0], [24.0, 29.0], [23.0, 29.0],
                                    [23.0, 30.0], [24.0, 30.0], [23.0, 30.0], [22.0, 30.0], [23.0, 30.0],
                                    [22.0, 30.0], [21.0, 30.0], [20.0, 30.0], [19.0, 30.0], [19.0, 31.0],
                                    [19.0, 32.0], [19.0, 33.0], [19.0, 34.0], [19.0, 35.0], [19.0, 36.0],
                                    [19.0, 37.0], [19.0, 38.0], [19.0, 39.0], [18.0, 39.0],
                                    [17.0, 39.0]], dtype=tf.float64),
                    4: tf.constant([[16.0, 39.0], [15.0, 39.0], [14.0, 39.0], [13.0, 39.0], [12.0, 39.0],
                                    [12.0, 40.0], [12.0, 41.0], [12.0, 42.0], [12.0, 43.0], [12.0, 44.0],
                                    [12.0, 45.0], [12.0, 46.0], [12.0, 47.0], [12.0, 48.0], [11.0, 48.0],
                                    [12.0, 48.0]], dtype=tf.float64),
                    -1: tf.constant([[11.0, 48.0], [10.0, 48.0], [9.0, 48.0], [8.0, 48.0], [7.0, 48.0],
                                     [6.0, 48.0], [5.0, 48.0], [4.0, 48.0], [3.0, 48.0], [2.0, 48.0],
                                     [2.0, 47.0], [2.0, 46.0], [2.0, 45.0], [2.0, 44.0], [2.0, 43.0],
                                     [2.0, 42.0], [2.0, 41.0], [2.0, 40.0], [2.0, 39.0], [2.0, 38.0],
                                     [2.0, 37.0], [2.0, 36.0]], dtype=tf.float64)}

uav_2_trajectory = {6: tf.constant([[25.0, 25.0], [26.0, 25.0], [27.0, 25.0], [28.0, 25.0], [29.0, 25.0],
                                    [30.0, 25.0], [31.0, 25.0], [32.0, 25.0], [35.0, 25.0], [40.0, 26.0],
                                    [40.0, 27.0]], dtype=tf.float64),
                    7: tf.constant([[33.0, 25.0], [34.0, 25.0], [36.0, 25.0], [37.0, 25.0],
                                    [38.0, 25.0], [39.0, 25.0], [40.0, 25.0], [41.0, 27.0]], dtype=tf.float64),
                    9: tf.constant([[41.0, 26.0], [42.0, 26.0], [42.0, 25.0], [43.0, 25.0], [44.0, 25.0],
                                    [44.0, 24.0], [44.0, 23.0], [45.0, 23.0], [44.0, 22.0]], dtype=tf.float64),
                    8: tf.constant([[44.0, 21.0], [44.0, 20.0], [44.0, 19.0], [44.0, 18.0], [44.0, 17.0], [44.0, 16.0],
                                    [44.0, 15.0], [44.0, 14.0], [44.0, 13.0], [44.0, 12.0], [44.0, 11.0], [44.0, 10.0],
                                    [44.0, 9.0]], dtype=tf.float64),
                    -1: tf.constant([[44.0, 8.0], [44.0, 7.0], [44.0, 6.0], [44.0, 5.0], [44.0, 4.0],
                                     [44.0, 3.0], [44.0, 2.0], [43.0, 2.0], [42.0, 2.0], [41.0, 2.0],
                                     [40.0, 2.0], [39.0, 2.0], [38.0, 2.0], [37.0, 2.0], [36.0, 2.0],
                                     [35.0, 2.0], [34.0, 2.0], [33.0, 2.0], [32.0, 2.0], [31.0, 2.0],
                                     [30.0, 2.0], [29.0, 2.0]], dtype=tf.float64)}

# uav_positions = {0: uav_0_trajectory}
# uav_positions = {0: uav_0_trajectory, 1: uav_1_trajectory}
uav_positions = {0: uav_0_trajectory, 1: uav_1_trajectory, 2: uav_2_trajectory}

"""
Configurations-III: Traffic generation model
"""
depl_env, rf, le_l, le_m, le_h = 'rural', n_uavs, 1, 10, 100
arrival_rates_l = {_k: _v * rf for _k, _v in arrival_rates_r.items()}
arrival_rates_m = {_k: _v * rf * le_m for _k, _v in arrival_rates_r.items()}
arrival_rates_h = {_k: _v * rf * le_h for _k, _v in arrival_rates_r.items()}

"""
Configurations-IV: Channel model
"""

'''
TODO: Change k1, k2, z1, and z2 according to the deployment environment
TODO: Change n_c according to the deployment environment (Verizon LTE/LTE-A/5G)
'''

if depl_env == 'rural':
    n_c, k1, k2, z1, z2, arrival_rates = 2, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_l
elif depl_env == 'suburban':
    n_c, k1, k2, z1, z2, arrival_rates = 4, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_m
else:
    n_c, k1, k2, z1, z2, arrival_rates = 10, 1.0, np.log(100) / 90.0, 9.61, 0.16, arrival_rates_h

bw, n_xu = 20e6, 1
bw_, num_req = bw / n_c, 1000
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


def gn_request(env, num, chs, trxs, serv_idx, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Nodes with multiple transceivers (1, 2, and 3 UAVs) | GN request
    """
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        ch_time = env.now
        ch_w_times.append(ch_time - arrival_time)

        trxs_ = trxs[serv_idx]
        x = np.argmin([max([0, len(_x.put_queue) + len(_x.users)]) for _x in trxs_])

        with trxs_[x].request() as req_:
            yield req_
            trx_time = env.now
            trx_w_times.append(trx_time - ch_time)
            w_times.append(trx_time - arrival_time)
            yield env.timeout(serv_times[num])


def arrivals(env, chs, trxs, n_r, arr, serv_idx, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Nodes with multiple transceivers (1, 2, and 3 UAVs) | Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request(env, num, chs, trxs, serv_idx,
                               ch_w_times, trx_w_times, w_times, serv_times))

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
            executor.submit(self.__evaluate_los_throughput, d, phi, r_los)
            executor.submit(self.__evaluate_nlos_throughput, d, r_nlos)

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


def multiple_uav_relays(payload_sizes, gn_alts, gn_coords, num_workers):
    gu_delays = {k: {_k: {} for _k in v.keys()} for k, v in uav_positions.items()}

    gu_xy_distances = {k: {_k: tf.norm(tf.subtract(gn_coords[_k] * scaling_factor, _v * scaling_factor), axis=1)
                           for _k, _v in v.items()} for k, v in uav_positions.items()}

    gu_heights = {k: {_k: tf.constant(abs(uav_height - gn_alts[_k]), shape=_v.shape, dtype=tf.float64)
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
                                              payload_sizes, num_workers).aggregated_average_delay

    return gu_delays


def evaluate(num_workers):
    delays = multiple_uav_relays(data_payload_sizes, gn_heights, gn_positions, num_workers)

    delays_mod = {p: {k: np.random.permutation(
        np.repeat([v[_k][p] for _k in v.keys()], num_req)) for k, v in delays.items()} for p in data_payload_sizes}

    for p, v in delays_mod.items():
        p_len = p / 1e6
        totals = {_u: 0.0 for _u in range(n_uavs)}
        services, waits = {_u: 0.0 for _u in range(n_uavs)}, {_u: 0.0 for _u in range(n_uavs)}
        ch_waits, trx_waits = {_u: 0.0 for _u in range(n_uavs)}, {_u: 0.0 for _u in range(n_uavs)}

        for _k, _v in v.items():
            _waits, _ch_waits, _trx_waits, e = [], [], [], Environment()

            e.process(arrivals(e, [Resource(e) for _ in range(n_c)],
                               {_u: [Resource(e) for _ in range(n_xu)] for _u in range(n_uavs)},
                               len(_v), arrival_rates[p], _k, _ch_waits, _trx_waits, _waits, _v))

            e.run()

            services[_k] = np.mean(_v)
            waits[_k] = np.mean(_waits)
            ch_waits[_k] = np.mean(_ch_waits)
            trx_waits[_k] = np.mean(_trx_waits)
            totals[_k] = services[_k] + waits[_k]

        print(f'[DEBUG] DDQNEvaluation evaluate: Payload Size = {p_len} Mb | '
              f'Average Wait Time (Channel) = {np.mean([_ for _ in ch_waits.values()])} seconds.')

        print(f'[DEBUG] DDQNEvaluation evaluate: Payload Size = {p_len} Mb | '
              f'Average Wait Time (Transceiver) = {np.mean([_ for _ in trx_waits.values()])} seconds.')

        print(f'[DEBUG] DDQNEvaluation evaluate: Payload Size = {p_len} Mb | '
              f'Average Communication Service Time = {np.mean([_ for _ in services.values()])} seconds.')

        print('[INFO] DDQNEvaluation evaluate: '
              f'[{n_uavs}] UAV Relays | Payload Length = {p_len} Mb | '
              f'UAV Power Consumption Constraint = {evaluate_power_consumption() / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean([_ for _ in totals.values()])} seconds.\n')


# Run Trigger
if __name__ == '__main__':
    evaluate(number_of_workers)
