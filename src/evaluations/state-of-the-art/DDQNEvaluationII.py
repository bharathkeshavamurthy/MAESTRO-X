"""
This script evaluates the performance of a DDQN implementation of the Multi-UAV Data Harvesting Problem.

Please see uav-data-harvesting git submodule for the code corresponding to this state-of-the-art framework. Also, check
out the results section in the parent repository (Odin) for screenshots and logs corresponding to the execution of this
framework on our ASU ECEE NVIDIA A100 GPU cluster.

DDQNSoAEvaluation v2.0: A different "more realistic" way of evaluating the framework's performance: consider penalties.

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

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logging level

import numpy as np
import tensorflow as tf
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Configurations: Deployment Model Parameters
"""

# The NumPy random seed for reproducibility...
np.random.seed(6)

# The scaling factor from "urban50" to "odin-uav-mobility"
scaling_factor = 20

# The request arrival rate from the GNs in this evaluation
arrival_rates = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}

# The data payload size from each GN (in bits) | Multiple evaluations
data_payload_sizes = [1e6, 10e6, 100e6]

# The number of UAVs in model training and in this post-processing evaluations
number_of_uavs = 1

# The UAV forward flying velocity for mobility power consumption analysis (Fixed in this framework) (in m/s)
uav_velocity = 10.0

# The GN heights extracted from the "urban50" scenario run on 4xNVIDIA-A100 GPU cluster at ASU School of ECEE
gn_heights = {0: tf.constant(0.0, dtype=tf.float64), 1: tf.constant(0.0, dtype=tf.float64),
              2: tf.constant(0.0, dtype=tf.float64), 3: tf.constant(0.0, dtype=tf.float64),
              4: tf.constant(0.0, dtype=tf.float64), 5: tf.constant(0.0, dtype=tf.float64),
              6: tf.constant(0.0, dtype=tf.float64), 7: tf.constant(0.0, dtype=tf.float64),
              8: tf.constant(0.0, dtype=tf.float64), 9: tf.constant(0.0, dtype=tf.float64)}

# The GN positions extracted from the "urban50" scenario run on 4xNVIDIA-A100 GPU cluster at ASU School of ECEE
gn_positions = {0: tf.constant([0.0, 9.0], dtype=tf.float64), 1: tf.constant([4.0, 14.0], dtype=tf.float64),
                2: tf.constant([9.0, 15.0], dtype=tf.float64), 3: tf.constant([14.0, 2.0], dtype=tf.float64),
                4: tf.constant([14.0, 50.0], dtype=tf.float64), 5: tf.constant([20.0, 43.0], dtype=tf.float64),
                6: tf.constant([42.0, 27.0], dtype=tf.float64), 7: tf.constant([42.0, 24.0], dtype=tf.float64),
                8: tf.constant([47.0, 8.0], dtype=tf.float64), 9: tf.constant([48.0, 22.0], dtype=tf.float64)}

# The height of all the UAVs during model training and during this post-processing evaluation (in meters)
uav_height = 200.0

"""
UAV Trajectories
"""

# The trajectory of UAV-0 to serve GNs 2, 1, 0, and 3 (approximated to the closest integer)
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
                                    [15.0, 5.0]], dtype=tf.float64)}

# The trajectory of UAV-1 to serve GNs 5 and 4 (approximated to the closest integer)
uav_1_trajectory = {5: tf.constant([[24.0, 26.0], [24.0, 27.0], [24.0, 28.0], [24.0, 29.0], [23.0, 29.0],
                                    [23.0, 30.0], [24.0, 30.0], [23.0, 30.0], [22.0, 30.0], [23.0, 30.0],
                                    [22.0, 30.0], [21.0, 30.0], [20.0, 30.0], [19.0, 30.0], [19.0, 31.0],
                                    [19.0, 32.0], [19.0, 33.0], [19.0, 34.0], [19.0, 35.0], [19.0, 36.0],
                                    [19.0, 37.0], [19.0, 38.0], [19.0, 39.0], [18.0, 39.0],
                                    [17.0, 39.0]], dtype=tf.float64),
                    4: tf.constant([[16.0, 39.0], [15.0, 39.0], [14.0, 39.0], [13.0, 39.0], [12.0, 39.0],
                                    [12.0, 40.0], [12.0, 41.0], [12.0, 42.0], [12.0, 43.0], [12.0, 44.0],
                                    [12.0, 45.0], [12.0, 46.0], [12.0, 47.0], [12.0, 48.0], [11.0, 48.0],
                                    [12.0, 48.0]], dtype=tf.float64)}

# The trajectory of UAV-2 to serve GNs 6, 7, 9, and 8 (approximated to the closest integer)
uav_2_trajectory = {6: tf.constant([[25.0, 25.0], [26.0, 25.0], [27.0, 25.0], [28.0, 25.0], [29.0, 25.0],
                                    [30.0, 25.0], [31.0, 25.0], [32.0, 25.0], [35.0, 25.0], [40.0, 26.0],
                                    [40.0, 27.0]], dtype=tf.float64),
                    7: tf.constant([[33.0, 25.0], [34.0, 25.0], [36.0, 25.0], [37.0, 25.0],
                                    [38.0, 25.0], [39.0, 25.0], [40.0, 25.0], [41.0, 27.0]], dtype=tf.float64),
                    9: tf.constant([[41.0, 26.0], [42.0, 26.0], [42.0, 25.0], [43.0, 25.0], [44.0, 25.0],
                                    [44.0, 24.0], [44.0, 23.0], [45.0, 23.0], [44.0, 22.0]], dtype=tf.float64),
                    8: tf.constant([[44.0, 21.0], [44.0, 20.0], [44.0, 19.0], [44.0, 18.0], [44.0, 17.0], [44.0, 16.0],
                                    [44.0, 15.0], [44.0, 14.0], [44.0, 13.0], [44.0, 12.0], [44.0, 11.0], [44.0, 10.0],
                                    [44.0, 9.0]], dtype=tf.float64)}

"""
The system-wide resource for UAV power consumption evaluation
"""


def evaluate_power_consumption(v=uav_velocity):
    u_tip, v_0, p_1, p_2, p_3 = 200.0, 7.2, 580.65, 790.6715, 0.0073
    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + \
           (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5) + \
           (p_3 * (v ** 3))


"""
The system-wide resources for [M/G/$N_{K}$] Queueing System evaluation [SimPy]
"""


def gn_request(env, num, chs, w_times, serv_times):
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])
    with chs[k].request() as req:
        yield req
        w_times.append(env.now - arrival_time)
        yield env.timeout(serv_times[num])


def arrivals(env, chs, n_r, arr, w_times, serv_times):
    for num in range(n_r):
        env.process(gn_request(env, num, chs, w_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


"""
The system-wide resource for [Link Performance] evaluation
"""


class LinkPerformanceEvaluator(object):

    def __init__(self, channel_bandwidth, reference_snr_at_1_meter, los_path_loss_exponent,
                 nlos_path_loss_exponent, nlos_attenuation_constant, los_rician_factor_1, los_rician_factor_2,
                 propagation_environment_parameter_1, propagation_environment_parameter_2,
                 convergence_confidence, bisection_method_tolerance):
        self.channel_bandwidth = channel_bandwidth
        self.reference_snr_at_1_meter = reference_snr_at_1_meter
        self.los_path_loss_exponent = los_path_loss_exponent
        self.nlos_path_loss_exponent = nlos_path_loss_exponent
        self.nlos_attenuation_constant = nlos_attenuation_constant
        self.los_rician_factor_1 = los_rician_factor_1
        self.los_rician_factor_2 = los_rician_factor_2
        self.propagation_environment_parameter_1 = propagation_environment_parameter_1
        self.propagation_environment_parameter_2 = propagation_environment_parameter_2
        self.convergence_confidence = convergence_confidence
        self.bisection_method_tolerance = bisection_method_tolerance

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
        f_z, q_m = self.__f_z(z), self.__marcum_q(df, nc, (y * (z ** 2)))
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
        return ((2 ** (gamma / b)) - 1) / (gamma_ * (lambda: kappa, lambda: 1)[los]() *
                                           (d ** (lambda: -alpha_, lambda: -alpha)[los]()))

    def __evaluate_los_throughput(self, d, phi, r_los):
        k_1, k_2 = self.los_rician_factor_1, self.los_rician_factor_2
        k, alpha = k_1 * np.exp(k_2 * phi), self.los_path_loss_exponent
        b, gamma_ = self.channel_bandwidth, self.reference_snr_at_1_meter
        df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), self.bisection_method_tolerance
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) *
                                                                            gamma_ * (d ** -alpha))), t)
        tf.compat.v1.assign(r_los, self.__f_z(z_star), use_locking=True)

    def __evaluate_nlos_throughput(self, d, r_nlos):
        alpha_, t = self.nlos_path_loss_exponent, self.bisection_method_tolerance
        b, gamma_, kappa = self.channel_bandwidth, self.reference_snr_at_1_meter, self.nlos_attenuation_constant
        df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) *
                                                                            gamma_ * kappa * (d ** -alpha_))), t)
        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), use_locking=True)

    def __calculate_adapted_throughput(self, d, phi, r_los, r_nlos, num_workers):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.submit(self.__evaluate_los_throughput, d, phi, r_los)
            executor.submit(self.__evaluate_nlos_throughput, d, r_nlos)

    def average_throughputs(self, d_s, phi_s, num_workers):
        r_los_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)
        z_1, z_2 = self.propagation_environment_parameter_1, self.propagation_environment_parameter_1
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, (d, phi) in enumerate(zip(d_s, phi_s)):
                executor.submit(self.__calculate_adapted_throughput, d, phi, r_los_s[i], r_nlos_s[i], num_workers)
        phi_degrees = (180.0 / np.pi) * phi_s
        p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
        p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)
        return tf.add(tf.multiply(p_los, r_los_s), tf.multiply(p_nlos, r_nlos_s))

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformanceEvaluator Termination: Tearing things down - Exception Type = {exc_type} | '
              f'Exception Value = {exc_val} | Traceback = {exc_tb}')


"""
Core Evaluation Routines
"""


def multiple_uav_relays(payload_sizes, uav_trajs, gn_alts, gn_coords, num_workers):
    gu_delays = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    gu_powers = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    p_extra = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    gu_time_penalties = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    gu_energy_penalties = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    gu_throughputs = {k: {_k: {} for _k in v.keys()} for k, v in uav_trajs.items()}
    gu_times = {k: {_k: tf.divide(tf.norm(tf.roll(_v * scaling_factor, shift=-1, axis=0)[:-1, :] -
                                          _v[:-1, :] * scaling_factor, axis=1), uav_velocity)
                    for _k, _v in v.items()} for k, v in uav_trajs.items()}
    gu_xy_distances = {k: {_k: tf.norm(tf.subtract(gn_coords[_k] * scaling_factor, _v * scaling_factor), axis=1)
                           for _k, _v in v.items()} for k, v in uav_trajs.items()}
    gu_heights = {k: {_k: tf.constant(abs(uav_height - gn_alts[_k]), shape=_v.shape, dtype=tf.float64)
                      for _k, _v in v.items()} for k, v in gu_xy_distances.items()}
    gu_distances = {k: {_k: tf.sqrt(tf.add(tf.square(gu_xy_distances[k][_k]), tf.square(gu_heights[k][_k])))
                        for _k in v.keys()} for k, v in gu_xy_distances.items()}
    gu_angles = {k: {_k: tf.asin(tf.divide(gu_heights[k][_k], gu_distances[k][_k]))
                     for _k in v.keys()} for k, v in gu_xy_distances.items()}
    lpf = LinkPerformanceEvaluator(5e6, 1e4, 2.0, 2.8, 0.2, 1.0, np.log(100) / 90.0, 9.61, 0.16, 100, 1e-10)
    for k, v in gu_throughputs.items():
        for _k, _v in v.items():
            dists, angles = gu_distances[k][_k], gu_angles[k][_k]
            gu_throughputs[k][_k] = lpf.average_throughputs(dists, angles, num_workers)
            p_extra[k][_k] = {p_len: tf.subtract(p_len, tf.reduce_sum(tf.multiply(gu_times[k][_k],
                                                                                  gu_throughputs[k][_k][:-1]), axis=0))
                              for p_len in payload_sizes}
            gu_time_penalties[k][_k] = {
                p_len: p_extra[k][_k][p_len] / gu_throughputs[k][_k][-1] if p_extra[k][_k][p_len] > 0.0 else 0.0
                for p_len in payload_sizes}
            gu_delays[k][_k] = {p_len: tf.add(tf.reduce_sum(gu_times[k][_k], axis=0), gu_time_penalties[k][_k][p_len])
                                for p_len in payload_sizes}
            gu_energy_penalties[k][_k] = {p_len: evaluate_power_consumption(0.0) * gu_time_penalties[k][_k][p_len]
                                          for p_len in payload_sizes}
            gu_powers[k][_k] = {p_len: tf.divide(tf.add(gu_energy_penalties[k][_k][p_len],
                                                        tf.multiply(evaluate_power_consumption(),
                                                                    tf.reduce_sum(gu_times[k][_k]))),
                                                 gu_delays[k][_k][p_len]) for p_len in payload_sizes}
    return gu_delays, gu_powers


def evaluate_operations(num_workers=256):
    uav_positions = {0: uav_0_trajectory}
    delays, powers = multiple_uav_relays(data_payload_sizes, uav_positions, gn_heights, gn_positions, num_workers)
    powers_mod = {p: {k: np.repeat([v[_k][p].numpy() for _k in v.keys()], 10000)
                      for k, v in powers.items()} for p in data_payload_sizes}
    delays_mod = {p: {k: np.random.permutation(np.repeat([v[_k][p].numpy() for _k in v.keys()], 10000))
                      for k, v in delays.items()} for p in data_payload_sizes}
    for p, v in delays_mod.items():
        p_len = p / 1e6
        services, waits, totals = {0: 0.0, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 0.0}, {0: 0.0, 1: 0.0, 2: 0.0}
        for _k, _v in v.items():
            _waits, e = [], Environment()
            e.process(arrivals(e, [Resource(e)], len(_v), arrival_rates[p], _waits, _v))
            e.run()
            services[_k] = np.mean(_v)
            waits[_k] = np.mean(_waits)
            totals[_k] = services[_k] + waits[_k]
        print(f'[DEBUG] DDQNEvaluationII evaluate_operations: Payload Size = {p_len} Mb | '
              f'Average Comm Delay = {np.mean([_ for _ in services.values()])} seconds')
        print(f'[DEBUG] DDQNEvaluationII evaluate_operations: Payload Size = {p_len} Mb | '
              f'Average Wait Times = {np.mean([_ for _ in waits.values()])} seconds')
        print(f'[INFO] DDQNEvaluationII evaluate_operations: Multiple UAV Relays | M/G/1 | '
              f'[{number_of_uavs}] UAV Relays | Payload Length = {p_len} Mb | '
              f'Average Service Delay = {np.mean([_ for _ in totals.values()])} seconds | '
              f'UAV Power Consumption = {np.mean(np.concatenate([_ for _ in powers_mod[p].values()]))} Watts\n')


# Run Trigger
if __name__ == '__main__':
    evaluate_operations()
