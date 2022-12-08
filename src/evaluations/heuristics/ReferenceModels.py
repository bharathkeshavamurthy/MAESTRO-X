"""
1. This script evaluates the performance of the BS alone servicing GN requests.
2. This script evaluates the performance of HAPs at a height of $H_{P}$ to serve GN requests.
3. This script evaluates the performance of an LEO constellation at a height of $H_{L}$ to serve GN requests.
4. This script evaluates the performance of static UAVs hovering at a fixed height $H_{U}$ to serve GN requests.

The key metric analyzed in this is:

    "UAV Average Power Constraint (in Watts) v GN Active Communication Request Delay (in seconds)";
    for [BS/HAP/LEO serving the GNs] and for [Number of UAVs ($N_{U}$) = 1, 2, 3, 5, 10];
    for Payload Lengths ($L$) = 1.0 Mb, 10.0 Mb, and 100.0 Mb.

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

# Numpy seed
np.random.seed(6)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
CONFIGURATIONS-II: Simulation parameters
"""

''' Deployment model '''

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The height of the HAP from the ground ($H_{U}$) in meters
HAP_HEIGHT = 5e6

# The height of the LEO from the ground ($H_{U}$) in meters
LEO_HEIGHT = 250e6

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1000.0

# The total number of GNs (implies communication requests) in the cell under analysis
NUMBER_OF_REQUESTS = int(1e4)

# The number of communication requests originating in the cell per second ($\Lambda$) in requests/second
# 1.0 Mb: One request every minute | 10.0 Mb: One request every 5 minutes | 100.0 Mb: One request every 30 minutes
ARRIVAL_RATES = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.5555e-4}

''' UAV mobility power consumption model '''

# The mean rotor-induced velocity ($v_{0}$) in m/s
MEAN_ROTOR_INDUCED_VELOCITY = 7.2

# The maximum UAV velocity ($V_{\text{max}}$) in m/s
MAX_UAV_VELOCITY = 55.0

# The rotor blade tip speed ($U_{\text{tip}}$) in m/s
ROTOR_BLADE_TIP_SPEED = 200.0

# The UAV power consumption profile constant 1 ($P_{1}$) relevant for blade profile evaluation
POWER_PROFILE_CONSTANT_1 = 580.65

# The UAV power consumption profile constant 2 ($P_{2}$) relevant for induced velocity profile evaluation
POWER_PROFILE_CONSTANT_2 = 790.6715

# The UAV power consumption profile constant 3 ($P_{3}$) which corresponds to a parasite term in the model
POWER_PROFILE_CONSTANT_3 = 0.0073

# The thrust-to-weight ratio in the rotary-wing UAV motion model ($\kappa_{\text{UAV}}{\triangleq}\frac{T}{W}$
THRUST_TO_WEIGHT_RATIO = 1.0

''' Channel model '''

# The total FCC-allocated bandwidth for this application ($B$) in Hz
TOTAL_BANDWIDTH = 20e6

# The number of orthogonal data channels ($N_{C}$) in this deployment
NUMBER_OF_CHANNELS = 4

# The bandwidth available per orthogonal data channel ($B_{k}$) in Hz
CHANNEL_BANDWIDTH = TOTAL_BANDWIDTH / NUMBER_OF_CHANNELS

# The number of transceivers per UAV in our deployment ($N_{X|U}$)
NUMBER_OF_TRANSCEIVERS_UAV = 4

# The number of transceivers at the BS in our deployment ($N_{X|B}$)
NUMBER_OF_TRANSCEIVERS_BS = 10

# The number of transceivers at the HAP in our deployment ($N_{X|P}$)
NUMBER_OF_TRANSCEIVERS_HAP = 10

# The number of transceivers at the LEO in our deployment ($N_{X|L}$)
NUMBER_OF_TRANSCEIVERS_LEO = 10

# The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$)
REFERENCE_SNR_AT_1_METER = linear((5e6 * 40) / CHANNEL_BANDWIDTH)

# The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
LoS_PATH_LOSS_EXPONENT = 2.0

# The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
NLoS_PATH_LOSS_EXPONENT = 2.8

# The propagation environment specific parameter ($z_{1}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61

# The propagation environment specific parameter ($z_{2}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16

# The propagation environment dependent coefficient ($k_{1}$) for the LoS Rician link model's $K$-factor
LoS_RICIAN_FACTOR_1 = 1.0

# The propagation environment dependent coefficient ($k_{2}$) for the LoS Rician link model's $K$-factor
LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0

# The additional attenuation constant for NLoS links ($\kappa$) | This factor affects the large-scale fading dynamics
NLoS_ATTENUATION_CONSTANT = 0.2

''' Algorithmic model '''

# The convergence confidence level for optimization algorithms in this framework
CONVERGENCE_CONFIDENCE = 10

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

"""
Utilities
"""


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
        self.reference_snr_at_1_meter = reference_snr_at_1_meter
        self.nlos_attenuation_constant = nlos_attenuation_constant
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
    def __average_delays(self, p_lens, r_bars):
        delta_bars_p = {
            p_len: tf.divide(tf.constant(p_len, shape=r_bars.shape, dtype=tf.float64), r_bars) for p_len in p_lens
        }

        delta_bars_agg = {
            p_len: tf.divide(tf.reduce_sum(delta_bars), tf.constant(r_bars.shape[0], dtype=tf.float64)).numpy()
            for p_len, delta_bars in delta_bars_p.items()
        }

        return delta_bars_p, delta_bars_agg

    def evaluate(self, d_s, phi_s, p_lens, num_workers):
        r_los_s, r_nlos_s, r_bars = self.__average_throughputs(d_s, phi_s, num_workers)

        delta_bars_p, delta_bars_agg = self.__average_delays(p_lens, r_bars)
        return self.evaluation_output(aggregated_average_delay=delta_bars_agg,
                                      los_throughputs=r_los_s, nlos_throughputs=r_nlos_s,
                                      average_throughputs=r_bars, average_delays=delta_bars_p)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformance Termination: Tearing things down - '
              f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {exc_tb}.')


def evaluate_power_consumption(uav_flying_velocity):
    """
    UAV mobility power consumption
    """
    v, u_tip, v_0 = uav_flying_velocity, ROTOR_BLADE_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY
    p_1, p_2, p_3 = POWER_PROFILE_CONSTANT_1, POWER_PROFILE_CONSTANT_2, POWER_PROFILE_CONSTANT_3

    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_3 * (v ** 3)) + \
        (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)


static_uav_pavg = evaluate_power_consumption(0.0)


def gn_request(env, num, chs, trxs, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Single node with multiple transceivers (BS/HAP/LEO/1-UAV) | GN request
    """
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        ch_time = env.now
        ch_w_times.append(ch_time - arrival_time)

        x = np.argmin([max([0, len(_x.put_queue) + len(_x.users)]) for _x in trxs])

        with trxs[x].request() as req_:
            yield req_
            trx_time = env.now
            trx_w_times.append(trx_time - ch_time)
            w_times.append(trx_time - arrival_time)
            yield env.timeout(serv_times[num])


def arrivals(env, chs, trxs, n_r, arr, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Single node with multiple transceivers (BS/HAP/LEO/1-UAV) | Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request(env, num, chs, trxs, ch_w_times,
                               trx_w_times, w_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


def gn_request_(env, num, chs, trxs, serv_idxs, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Multiple nodes with multiple transceivers (2, 3, 5, 10 UAVs) | GN request
    """
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        ch_time = env.now
        ch_w_times.append(ch_time - arrival_time)

        trxs_ = trxs[serv_idxs[num].numpy()]
        x = np.argmin([max([0, len(_x.put_queue) + len(_x.users)]) for _x in trxs_])

        with trxs_[x].request() as req_:
            yield req_
            trx_time = env.now
            trx_w_times.append(trx_time - ch_time)
            w_times.append(trx_time - arrival_time)
            yield env.timeout(serv_times[num])


def arrivals_(env, chs, trxs, n_r, arr, serv_idxs, ch_w_times, trx_w_times, w_times, serv_times):
    """
    Simpy queueing model: Multiple nodes with multiple transceivers (2, 3, 5, 10 UAVs) | Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request_(env, num, chs, trxs, serv_idxs,
                                ch_w_times, trx_w_times, w_times, serv_times))

        yield env.timeout(-np.log(np.random.random_sample()) / arr)


"""
Core operations
"""


def bs_only(payload_lengths, bs_coords, gn_coords, num_workers):
    gb_xy_distances = tf.norm(tf.subtract(gn_coords, bs_coords), axis=1)
    gb_heights = tf.constant(abs(BASE_STATION_HEIGHT - 0.0), shape=gb_xy_distances.shape, dtype=tf.float64)

    gb_distances = tf.sqrt(tf.add(tf.square(gb_xy_distances), tf.square(gb_heights)))
    gb_angles = tf.asin(tf.divide(gb_heights, gb_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gb_delays = link_performance.evaluate(gb_distances, gb_angles, payload_lengths, num_workers).average_delays

    return gb_delays


def hap_only(payload_lengths, hap_coords, gn_coords, num_workers):
    gh_xy_distances = tf.norm(tf.subtract(gn_coords, hap_coords), axis=1)
    gh_heights = tf.constant(abs(HAP_HEIGHT - 0.0), shape=gh_xy_distances.shape, dtype=tf.float64)

    gh_distances = tf.sqrt(tf.add(tf.square(gh_xy_distances), tf.square(gh_heights)))
    gh_angles = tf.asin(tf.divide(gh_heights, gh_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gh_delays = link_performance.evaluate(gh_distances, gh_angles, payload_lengths, num_workers).average_delays

    return gh_delays


def leo_only(payload_lengths, leo_coords, gn_coords, num_workers):
    gl_xy_distances = tf.norm(tf.subtract(gn_coords, leo_coords), axis=1)
    gl_heights = tf.constant(abs(BASE_STATION_HEIGHT - 0.0), shape=gl_xy_distances.shape, dtype=tf.float64)

    gl_distances = tf.sqrt(tf.add(tf.square(gl_xy_distances), tf.square(gl_heights)))
    gl_angles = tf.asin(tf.divide(gl_heights, gl_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gl_delays = link_performance.evaluate(gl_distances, gl_angles, payload_lengths, num_workers).average_delays

    return gl_delays


def single_uav_relay(payload_lengths, bs_coords, uav_coords, gn_coords, num_workers):
    # GN-to-UAV links

    gu_xy_distances = tf.norm(tf.subtract(gn_coords, uav_coords), axis=1)
    gu_heights = tf.constant(abs(UAV_HEIGHT - 0.0), shape=gu_xy_distances.shape, dtype=tf.float64)

    gu_distances = tf.sqrt(tf.add(tf.square(gu_xy_distances), tf.square(gu_heights)))
    gu_angles = tf.asin(tf.divide(gu_heights, gu_distances))

    # UAV-to-BS links

    ub_xy_distances = tf.norm(tf.subtract(uav_coords, bs_coords), axis=1)
    ub_heights = tf.constant(abs(UAV_HEIGHT - BASE_STATION_HEIGHT), shape=ub_xy_distances.shape, dtype=tf.float64)

    ub_distances = tf.sqrt(tf.add(tf.square(ub_xy_distances), tf.square(ub_heights)))
    ub_angles = tf.asin(tf.divide(ub_heights, ub_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gu_delays = link_performance.evaluate(gu_distances, gu_angles, payload_lengths, num_workers).average_delays
        ub_delays = link_performance.evaluate(ub_distances, ub_angles, payload_lengths, num_workers).average_delays

    return {p_len: tf.add(gu_delays[p_len], ub_delays[p_len]) for p_len in payload_lengths}


def multiple_uav_relays(payload_lengths, bs_coords, multiple_uav_coords, gn_coords, num_workers):
    # GN-to-UAV links

    gu_xy_distances = [tf.norm(tf.subtract(gn_coords, uav_coords), axis=1) for uav_coords in multiple_uav_coords]
    gu_heights = tf.constant(abs(UAV_HEIGHT - 0.0), shape=gu_xy_distances[0].shape, dtype=tf.float64)

    gu_distances = tf.concat([tf.expand_dims(tf.sqrt(tf.add(
        tf.square(gu_xy_dists), tf.square(gu_heights))), axis=1) for gu_xy_dists in gu_xy_distances], axis=1)

    gu_angles = tf.concat([tf.expand_dims(tf.asin(tf.divide(
        gu_heights, gu_distances[:, idx])), axis=1) for idx in range(len(gu_xy_distances))], axis=1)

    min_indices = tf.math.argmin(gu_distances, axis=1)
    gu_distances_min = tf.reduce_min(gu_distances, axis=1)
    gu_angles_min = tf.constant([gu_angles[i, min_indices[i]].numpy()
                                 for i in range(min_indices.shape[0])], dtype=tf.float64)

    # UAV-to-BS links

    ub_xy_distances = tf.constant([tf.norm(tf.subtract(
        multiple_uav_coords[i], bs_coords), axis=1)[0].numpy() for i in min_indices.numpy()], dtype=tf.float64)
    ub_heights = tf.constant(abs(UAV_HEIGHT - BASE_STATION_HEIGHT), shape=ub_xy_distances.shape, dtype=tf.float64)

    ub_distances_min = tf.sqrt(tf.add(tf.square(ub_xy_distances), tf.square(ub_heights)))
    ub_angles_min = tf.asin(tf.divide(ub_heights, ub_distances_min))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gu_delays = link_performance.evaluate(gu_distances_min, gu_angles_min,
                                              payload_lengths, num_workers).average_delays
        ub_delays = link_performance.evaluate(ub_distances_min, ub_angles_min,
                                              payload_lengths, num_workers).average_delays

    return {p_len: (min_indices, tf.add(gu_delays[p_len], ub_delays[p_len])) for p_len in payload_lengths}


def simulate_ops(num_workers):
    number_of_gn_requests, payload_lengths, gn_coords = NUMBER_OF_REQUESTS, [1e6, 10e6, 100e6], []

    radii = np.random.uniform(0, CELL_RADIUS ** 2, number_of_gn_requests) ** 0.5
    angles = np.random.uniform(0, 2 * np.pi, number_of_gn_requests)

    gn_coords = tf.constant(list(zip(radii * np.cos(angles), radii * np.sin(angles))), dtype=tf.float64)

    ''' BS only '''

    bs_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                   dtype=tf.float64), axis=0), multiples=[number_of_gn_requests, 1])

    comm_delays_dict = bs_only(payload_lengths, bs_coords, gn_coords, num_workers)

    for payload_length, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, ch_waiting_times, trx_waiting_times = [], [], []
        comm_delays, environment = comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(
            environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
            [Resource(environment) for _ in range(NUMBER_OF_TRANSCEIVERS_BS)], number_of_gn_requests,
            ARRIVAL_RATES[payload_length], ch_waiting_times, trx_waiting_times, waiting_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels BS simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(comm_delays)} seconds.\n')

        print('[DEBUG] ReferenceModels BS simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Channel) = {np.mean(ch_waiting_times)} seconds.\n')

        print('[DEBUG] ReferenceModels BS simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Transceiver) = {np.mean(trx_waiting_times)} seconds.\n')

        print('[INFO] ReferenceModels BS simulate_ops: All requests served by the BS | '
              f'No UAV Relay | M/G/{NUMBER_OF_CHANNELS} and M/G/{NUMBER_OF_TRANSCEIVERS_BS} | '
              f'Payload Length = [{payload_length / 1e6}] Mb | Avg UAV Power Consumption Constraint = [N/A] kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, comm_delays))} seconds.\n')

    ''' HAP only '''

    hap_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                    dtype=tf.float64), axis=0), multiples=[number_of_gn_requests, 1])

    comm_delays_dict = hap_only(payload_lengths, hap_coords, gn_coords, num_workers)

    for payload_length, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, ch_waiting_times, trx_waiting_times = [], [], []
        comm_delays, environment = comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(
            environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
            [Resource(environment) for _ in range(NUMBER_OF_TRANSCEIVERS_HAP)], number_of_gn_requests,
            ARRIVAL_RATES[payload_length], ch_waiting_times, trx_waiting_times, waiting_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels HAP simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(comm_delays)} seconds.\n')

        print('[DEBUG] ReferenceModels HAP simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Channel) = {np.mean(ch_waiting_times)} seconds.\n')

        print('[DEBUG] ReferenceModels HAP simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Transceiver) = {np.mean(trx_waiting_times)} seconds.\n')

        print('[INFO] ReferenceModels HAP simulate_ops: All requests served by the HAP | '
              f'No UAV Relay | M/G/{NUMBER_OF_CHANNELS} and M/G/{NUMBER_OF_TRANSCEIVERS_HAP} | '
              f'Payload Length = [{payload_length / 1e6}] Mb | Avg UAV Power Consumption Constraint = [N/A] kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, comm_delays))} seconds.\n')

    ''' LEO only '''

    leo_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                    dtype=tf.float64), axis=0), multiples=[number_of_gn_requests, 1])

    comm_delays_dict = leo_only(payload_lengths, leo_coords, gn_coords, num_workers)

    for payload_length, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, ch_waiting_times, trx_waiting_times = [], [], []
        comm_delays, environment = comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(
            environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
            [Resource(environment) for _ in range(NUMBER_OF_TRANSCEIVERS_LEO)], number_of_gn_requests,
            ARRIVAL_RATES[payload_length], ch_waiting_times, trx_waiting_times, waiting_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels LEO simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(comm_delays)} seconds.\n')

        print('[DEBUG] ReferenceModels LEO simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Channel) = {np.mean(ch_waiting_times)} seconds.\n')

        print('[DEBUG] ReferenceModels LEO simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Transceiver) = {np.mean(trx_waiting_times)} seconds.\n')

        print('[INFO] ReferenceModels LEO simulate_ops: All requests served by the LEO | '
              f'No UAV Relay | M/G/{NUMBER_OF_CHANNELS} and M/G/{NUMBER_OF_TRANSCEIVERS_LEO} | '
              f'Payload Length = [{payload_length / 1e6}] Mb | Avg UAV Power Consumption Constraint = [N/A] kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, comm_delays))} seconds.\n')

    ''' 1 UAV-relay '''

    uav_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                    dtype=tf.float64), axis=0), multiples=[number_of_gn_requests, 1])

    comm_delays_dict = single_uav_relay(payload_lengths, bs_coords, uav_coords, gn_coords, num_workers)

    for payload_length, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, ch_waiting_times, trx_waiting_times = [], [], []
        comm_delays, environment = comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(
            environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
            [Resource(environment) for _ in range(NUMBER_OF_TRANSCEIVERS_UAV)], number_of_gn_requests,
            ARRIVAL_RATES[payload_length], ch_waiting_times, trx_waiting_times, waiting_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels UAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(comm_delays)} seconds.\n')

        print('[DEBUG] ReferenceModels UAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Channel) = {np.mean(ch_waiting_times)} seconds.\n')

        print('[DEBUG] ReferenceModels UAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Transceiver) = {np.mean(trx_waiting_times)} seconds.\n')

        print('[INFO] ReferenceModels UAV simulate_ops: Decode-Forward with UAV-relay | '
              f'Single UAV Relay | M/G/{NUMBER_OF_CHANNELS} and M/G/{NUMBER_OF_TRANSCEIVERS_UAV} | '
              f'Payload Length = [{payload_length / 1e6}] Mb | Avg Power Constraint = {static_uav_pavg / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, comm_delays))} seconds.\n')

    ''' Multiple UAV-relays '''

    """
    Configurations-III: UAV deployments for the 'Multiple UAV-relays' case
    """

    # 2 UAV-relays
    number_of_uavs = 2
    uav_1 = tf.constant([500.0, 0.0], dtype=tf.float64)
    uav_2 = tf.constant([-500.0, 0.0], dtype=tf.float64)
    multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gn_requests, 1])]

    """
    
    # 3 UAV-relays
    number_of_uavs = 3
    uav_1 = tf.constant([0.0, 500.0], dtype=tf.float64)
    uav_2 = tf.constant([500.0, -500.0], dtype=tf.float64)
    uav_3 = tf.constant([-500.0, -500.0], dtype=tf.float64)
    multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gn_requests, 1])]
    
    """

    """
    
    # 5 UAV-relays
    number_of_uavs = 5
    uav_1 = tf.constant([0.0, 500.0], dtype=tf.float64)
    uav_2 = tf.constant([500.0, 0.0], dtype=tf.float64)
    uav_3 = tf.constant([-500.0, 0.0], dtype=tf.float64)
    uav_4 = tf.constant([500.0, -500.0], dtype=tf.float64)
    uav_5 = tf.constant([-500.0, -500.0], dtype=tf.float64)
    multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_4, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_5, axis=0), multiples=[number_of_gn_requests, 1])]
                           
    """

    """
    
    # 10 UAV-relays
    number_of_uavs = 10
    uav_1 = tf.constant([0.0, 500.0], dtype=tf.float64)
    uav_2 = tf.constant([500.0, 0.0], dtype=tf.float64)
    uav_3 = tf.constant([0.0, -500.0], dtype=tf.float64)
    uav_4 = tf.constant([-500.0, 0.0], dtype=tf.float64)
    uav_5 = tf.constant([250.0, 250.0], dtype=tf.float64)
    uav_6 = tf.constant([500.0, -500.0], dtype=tf.float64)
    uav_7 = tf.constant([500.0, -250.0], dtype=tf.float64)
    uav_8 = tf.constant([-250.0, 250.0], dtype=tf.float64)
    uav_9 = tf.constant([-500.0, -500.0], dtype=tf.float64)
    uav_10 = tf.constant([-500.0, -250.0], dtype=tf.float64)
    multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_4, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_5, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_6, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_7, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_8, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_9, axis=0), multiples=[number_of_gn_requests, 1]),
                           tf.tile(tf.expand_dims(uav_10, axis=0), multiples=[number_of_gn_requests, 1])]
    
    """

    comm_delays_dict = multiple_uav_relays(payload_lengths, bs_coords, multiple_uav_coords, gn_coords, num_workers)

    for payload_length, comm_delays_tuple in comm_delays_dict.items():
        comm_delays_argmins, comm_delays_tensor = comm_delays_tuple
        waiting_times, ch_waiting_times, trx_waiting_times = [], [], []
        comm_delays, environment = comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals_(
            environment,
            [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
            {_u: Resource(environment) for _ in range(NUMBER_OF_TRANSCEIVERS_UAV) for _u in range(number_of_uavs)},
            number_of_gn_requests, ARRIVAL_RATES[payload_length], comm_delays_argmins,
            ch_waiting_times, trx_waiting_times, waiting_times, comm_delays)
        )

        environment.run()

        print('[DEBUG] ReferenceModels mUAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(comm_delays)} seconds.\n')

        print('[DEBUG] ReferenceModels mUAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Channel) = {np.mean(ch_waiting_times)} seconds.\n')

        print('[DEBUG] ReferenceModels mUAV simulate_ops: '
              f'Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay (Transceiver) = {np.mean(trx_waiting_times)} seconds.\n')

        print('[INFO] ReferenceModels mUAV simulate_ops: Decode-Forward with UAV-relays | '
              f'{number_of_uavs} UAV-relays | M/G/{NUMBER_OF_CHANNELS} and M/G/{NUMBER_OF_TRANSCEIVERS_UAV} | '
              f'Payload Length = [{payload_length / 1e6}] Mb | Per-UAV Avg Power = {static_uav_pavg / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, comm_delays))} seconds.')


# Run Trigger
if __name__ == '__main__':
    simulate_ops(num_workers=1024)
