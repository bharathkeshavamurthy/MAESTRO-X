"""
1. This script evaluates the performance of the BS alone servicing GN requests.
2. This script evaluates the performance of a HAP at a height of $H_{P}$ to serve GN requests.
3. This script evaluates the performance of static UAVs hovering at a fixed height $H_{U}$ to serve GN requests.

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
Configurations-II: Simulation parameters
"""

''' Deployment model '''

# The deployment under analysis
DEPLOYMENT_TYPE = 'BS'  # 'BS' | 'HAP' | 'UAV'

# The height of the BS from the ground ($H_{B}$) in meters
BS_HEIGHT = 80.0

# The height of the HAP from the ground ($H_{P}$) in meters
HAP_HEIGHT = 2e3

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The number of static UAVs in the deployment under analysis
NUMBER_OF_UAVS = 3

# The propagation environment under analysis
DEPLOYMENT_ENVIRONMENT = 'rural'  # 'rural', 'urban', 'suburban'

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1e3

# The smallest required distance between two GNs along the circumference of a specific radius level in m
MIN_CIRC_DISTANCE = 25.0

# The number of radii "levels" used in the discretization of comm state space $G_{R}+1$ or K_{R}+1$ or $N_{\text{sp}}$
RADII_LEVELS = 25

''' Traffic generation model '''

# The total number of GNs (implies communication requests) in the cell under analysis
NUMBER_OF_REQUESTS = int(1e4)

# The rate multiplication factor for a BS-deployment
BS_RATE_FACTOR = 3  # red (1x <=> 1-UAV) | green (2x <=> 2-UAVs) | blue (3x <=> 3-UAVs)

# The rate multiplication factor for a HAP-deployment
HAP_RATE_FACTOR = 3  # red (1x <=> 1-UAV) | green (2x <=> 2-UAVs) | blue (3x <=> 3-UAVs)

# The rate multiplication factor for a UAV-deployment
UAV_RATE_FACTOR = NUMBER_OF_UAVS  # red (1-UAV <=> 1x) | green (2-UAV <=> 2x) | blue (3-UAVs <=> 3x)

# The effective rate factor for the deployment under analysis
if DEPLOYMENT_TYPE == 'BS':
    RATE_FACTOR = BS_RATE_FACTOR
elif DEPLOYMENT_TYPE == 'HAP':
    RATE_FACTOR = HAP_RATE_FACTOR
else:
    RATE_FACTOR = UAV_RATE_FACTOR

# Raw arrival rates which will be scaled w.r.t the rate factor and the load escalation factor
RAW_ARRIVAL_RATES = {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}

# Low congestion ($\Lambda'$): 1.0 Mb: 5-reqs/1-min | 10.0 Mb: 1-req/1-min | 100.0 Mb: 1-req/6-min
LOAD_ESCALATION = 1
LOW_ARRIVAL_RATES = {_k: _v * RATE_FACTOR * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

# High congestion ($\Lambda'$): Load escalation factor * Arrival rates for the "low congestion" regime
LOAD_ESCALATION = 100
HIGH_ARRIVAL_RATES = {_k: _v * RATE_FACTOR * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

# Moderate congestion ($\Lambda'$): Load escalation factor * Arrival rates for the "moderate congestion" regime
LOAD_ESCALATION = 10
MODERATE_ARRIVAL_RATES = {_k: _v * RATE_FACTOR * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

# The effective arrival rates for the deployment under analysis
if DEPLOYMENT_ENVIRONMENT == 'rural':
    ARRIVAL_RATES = LOW_ARRIVAL_RATES
elif DEPLOYMENT_ENVIRONMENT == 'urban':
    ARRIVAL_RATES = HIGH_ARRIVAL_RATES
else:
    ARRIVAL_RATES = MODERATE_ARRIVAL_RATES

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

# The additional attenuation constant for NLoS links ($\kappa$)
NLoS_ATTENUATION_CONSTANT = 0.2

# The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
LoS_PATH_LOSS_EXPONENT = 2.0

'''
TODO: Change total_bandwidth and number_of_channels according to the deployment environment (Verizon LTE/LTE-A/5G)
'''
# The number of data channels in this deployment ($N_{C}$)
if DEPLOYMENT_ENVIRONMENT == 'rural':
    TOTAL_BANDWIDTH = 10e6
    NUMBER_OF_CHANNELS = 2  # Verizon rural: 2x 5-MHz LTE-A
elif DEPLOYMENT_ENVIRONMENT == 'urban':
    TOTAL_BANDWIDTH = 40e6
    NUMBER_OF_CHANNELS = 8  # Verizon urban: 8x 5-MHz LTE-A
else:
    TOTAL_BANDWIDTH = 20e6
    NUMBER_OF_CHANNELS = 4  # Verizon suburban: 4x 5-MHz LTE-A

# The bandwidth available per orthogonal data channel ($B$) in Hz
CHANNEL_BANDWIDTH = TOTAL_BANDWIDTH / NUMBER_OF_CHANNELS

# The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
NLoS_PATH_LOSS_EXPONENT = 2.8

'''
TODO: Change these propagation environment specific parameters according to the deployment environment
'''
# The propagation environment specific parameters used in our channel model
if DEPLOYMENT_ENVIRONMENT == 'rural':
    LoS_RICIAN_FACTOR_1 = 1.0
    LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0
    PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61
    PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16
elif DEPLOYMENT_ENVIRONMENT == 'urban':
    LoS_RICIAN_FACTOR_1 = 1.0
    LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0
    PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61
    PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16
else:
    LoS_RICIAN_FACTOR_1 = 1.0
    LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0
    PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61
    PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16

# The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$)
REFERENCE_SNR_AT_1_METER = linear((5e6 * 40) / CHANNEL_BANDWIDTH)

''' Algorithmic model '''

# The max number of concurrent workers allowed in this evaluation
NUMBER_OF_WORKERS = 1024

# The convergence confidence level for optimization algorithms in this framework
BISECTION_CONVERGENCE_CONFIDENCE = 10

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

"""
Configurations-III: Deployments (BS, HAP, GNs, and UAV(s))
"""

payload_sizes = [1e6, 10e6, 100e6]

''' GNs deployment (uniform-circular) '''

radii = np.linspace(start=0.0, stop=CELL_RADIUS, num=RADII_LEVELS)
angles = [np.linspace(start=0.0, stop=2 * np.pi, num=int((2 * np.pi * _r) / MIN_CIRC_DISTANCE) + 1) for _r in radii]

gn_coords = tf.concat([tf.constant(_r * np.einsum('ji', np.vstack([np.cos(angles[_i]),
                                                                   np.sin(angles[_i])])),
                                   dtype=tf.float64) for _i, _r in enumerate(radii)], axis=0)

number_of_gns = gn_coords.shape[0]
gn_indices = [_ for _ in range(number_of_gns)]

''' BS deployment '''
bs_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                               dtype=tf.float64), axis=0), multiples=[number_of_gns, 1])

''' HAP deployment '''
hap_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                dtype=tf.float64), axis=0), multiples=[number_of_gns, 1])

''' 1 static UAV-relay deployment '''
uav_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0],
                                                dtype=tf.float64), axis=0), multiples=[number_of_gns, 1])

''' Multiple static UAV-relays deployment '''

"""
# 2 static UAV-relays
number_of_uavs = NUMBER_OF_UAVS
uav_1 = tf.constant([500.0, 0.0], dtype=tf.float64)
uav_2 = tf.constant([-500.0, 0.0], dtype=tf.float64)
multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gns, 1])]
"""

# 3 static UAV-relays
number_of_uavs = NUMBER_OF_UAVS
uav_1 = tf.constant([0.0, 500.0], dtype=tf.float64)
uav_2 = tf.constant([500.0, -500.0], dtype=tf.float64)
uav_3 = tf.constant([-500.0, -500.0], dtype=tf.float64)
multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gns, 1])]

"""
# 5 static UAV-relays
number_of_uavs = NUMBER_OF_UAVS
uav_1 = tf.constant([0.0, 500.0], dtype=tf.float64)
uav_2 = tf.constant([500.0, 0.0], dtype=tf.float64)
uav_3 = tf.constant([-500.0, 0.0], dtype=tf.float64)
uav_4 = tf.constant([500.0, -500.0], dtype=tf.float64)
uav_5 = tf.constant([-500.0, -500.0], dtype=tf.float64)
multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_4, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_5, axis=0), multiples=[number_of_gns, 1])]
"""

"""
# 10 static UAV-relays
number_of_uavs = NUMBER_OF_UAVS
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
multiple_uav_coords = [tf.tile(tf.expand_dims(uav_1, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_2, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_3, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_4, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_5, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_6, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_7, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_8, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_9, axis=0), multiples=[number_of_gns, 1]),
                       tf.tile(tf.expand_dims(uav_10, axis=0), multiples=[number_of_gns, 1])]
"""

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


def gn_request(env, chs, w_times, s_times, serv_times):
    """
    Simpy queueing model: GN request
    """
    arr_time = env.now
    gn_num = np.random.choice(gn_indices)
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        w_times.append(env.now - arr_time)
        s_times.append(serv_times[gn_num])
        yield env.timeout(serv_times[gn_num])


def arrivals(env, chs, n_r, arr, w_times, s_times, serv_times):
    """
    Simpy queueing model: Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request(env, chs,
                               w_times, s_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


"""
Core operations
"""


def bs_only():
    gb_xy_distances = tf.norm(tf.subtract(gn_coords, bs_coords), axis=1)
    gb_heights = tf.constant(abs(BS_HEIGHT - 0.0), shape=gb_xy_distances.shape, dtype=tf.float64)

    gb_distances = tf.sqrt(tf.add(tf.square(gb_xy_distances), tf.square(gb_heights)))
    gb_angles = tf.asin(tf.divide(gb_heights, gb_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         BISECTION_CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gb_delays = link_performance.evaluate(gb_distances, gb_angles, payload_sizes, NUMBER_OF_WORKERS).average_delays

    return gb_delays


def hap_only():
    gh_xy_distances = tf.norm(tf.subtract(gn_coords, hap_coords), axis=1)
    gh_heights = tf.constant(abs(HAP_HEIGHT - 0.0), shape=gh_xy_distances.shape, dtype=tf.float64)

    gh_distances = tf.sqrt(tf.add(tf.square(gh_xy_distances), tf.square(gh_heights)))
    gh_angles = tf.asin(tf.divide(gh_heights, gh_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         BISECTION_CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gh_delays = link_performance.evaluate(gh_distances, gh_angles, payload_sizes, NUMBER_OF_WORKERS).average_delays

    return gh_delays


static_uav_pavg = evaluate_power_consumption(0.0)


def single_uav_relay():
    # GN-to-UAV links

    gu_xy_distances = tf.norm(tf.subtract(gn_coords, uav_coords), axis=1)
    gu_heights = tf.constant(abs(UAV_HEIGHT - 0.0), shape=gu_xy_distances.shape, dtype=tf.float64)

    gu_distances = tf.sqrt(tf.add(tf.square(gu_xy_distances), tf.square(gu_heights)))
    gu_angles = tf.asin(tf.divide(gu_heights, gu_distances))

    # UAV-to-BS links

    ub_xy_distances = tf.norm(tf.subtract(uav_coords, bs_coords), axis=1)
    ub_heights = tf.constant(abs(UAV_HEIGHT - BS_HEIGHT), shape=ub_xy_distances.shape, dtype=tf.float64)

    ub_distances = tf.sqrt(tf.add(tf.square(ub_xy_distances), tf.square(ub_heights)))
    ub_angles = tf.asin(tf.divide(ub_heights, ub_distances))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         BISECTION_CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gu_delays = link_performance.evaluate(gu_distances, gu_angles, payload_sizes, NUMBER_OF_WORKERS).average_delays
        ub_delays = link_performance.evaluate(ub_distances, ub_angles, payload_sizes, NUMBER_OF_WORKERS).average_delays

    return {p_len: tf.add(gu_delays[p_len], ub_delays[p_len]) for p_len in payload_sizes}


def multiple_uav_relays():
    # GN-to-UAV links

    gu_xy_distances = [tf.norm(tf.subtract(gn_coords, u_coords), axis=1) for u_coords in multiple_uav_coords]
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
    ub_heights = tf.constant(abs(UAV_HEIGHT - BS_HEIGHT), shape=ub_xy_distances.shape, dtype=tf.float64)

    ub_distances_min = tf.sqrt(tf.add(tf.square(ub_xy_distances), tf.square(ub_heights)))
    ub_angles_min = tf.asin(tf.divide(ub_heights, ub_distances_min))

    with LinkPerformance(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER,
                         LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT,
                         NLoS_ATTENUATION_CONSTANT, LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                         PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2,
                         BISECTION_CONVERGENCE_CONFIDENCE, BISECTION_METHOD_TOLERANCE) as link_performance:
        gu_delays = link_performance.evaluate(gu_distances_min, gu_angles_min,
                                              payload_sizes, NUMBER_OF_WORKERS).average_delays
        ub_delays = link_performance.evaluate(ub_distances_min, ub_angles_min,
                                              payload_sizes, NUMBER_OF_WORKERS).average_delays

    return {p_len: (min_indices, tf.add(gu_delays[p_len], ub_delays[p_len])) for p_len in payload_sizes}


def simulate_ops():
    """
    Simulate operations
    """

    ''' BS only '''

    comm_delays_dict = bs_only()

    for payload_size, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, service_times, comm_delays, environment = [], [], comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
                                     NUMBER_OF_REQUESTS, ARRIVAL_RATES[payload_size],
                                     waiting_times, service_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels BS simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(service_times)} seconds.')

        print('[DEBUG] ReferenceModels BS simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Wait Delay = {np.mean(waiting_times)} seconds.')

        print('[INFO] ReferenceModels BS simulate_ops: All requests served by BS | '
              f'No UAV Relay | M/G/{NUMBER_OF_CHANNELS} queuing at the data channels | '
              f'Payload Length = {payload_size / 1e6} Mb | UAV Power Consumption = N/A kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, service_times))} seconds.\n')

    ''' HAP only '''

    comm_delays_dict = hap_only()

    for payload_size, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, service_times, comm_delays, environment = [], [], comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
                                     NUMBER_OF_REQUESTS, ARRIVAL_RATES[payload_size],
                                     waiting_times, service_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels HAP simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(service_times)} seconds.')

        print('[DEBUG] ReferenceModels HAP simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Wait Delay = {np.mean(waiting_times)} seconds.')

        print('[INFO] ReferenceModels HAP simulate_ops: All requests served by HAP | '
              f'No UAV Relay | M/G/{NUMBER_OF_CHANNELS} queuing at the data channels | '
              f'Payload Length = {payload_size / 1e6} Mb | UAV Power Consumption = N/A kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, service_times))} seconds.\n')

    ''' 1 static UAV-relay '''

    comm_delays_dict = single_uav_relay()

    for payload_size, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, service_times, comm_delays, environment = [], [], comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
                                     NUMBER_OF_REQUESTS, ARRIVAL_RATES[payload_size],
                                     waiting_times, service_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels UAV simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(service_times)} seconds.')

        print('[DEBUG] ReferenceModels UAV simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Wait Delay = {np.mean(waiting_times)} seconds.')

        print('[INFO] ReferenceModels UAV simulate_ops: Decode-Forward with UAV-relay | '
              f'Single UAV Relay | M/G/{NUMBER_OF_CHANNELS} queuing at the data channels | '
              f'Payload Length = {payload_size / 1e6} Mb | UAV Power Consumption = {static_uav_pavg / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, service_times))} seconds.\n')

    ''' Multiple static UAV-relays '''

    comm_delays_dict = multiple_uav_relays()

    for payload_size, comm_delays_tuple in comm_delays_dict.items():
        comm_delays_argmins, comm_delays_tensor = comm_delays_tuple
        waiting_times, service_times, comm_delays, environment = [], [], comm_delays_tensor.numpy(), Environment()

        environment.process(arrivals(environment, [Resource(environment) for _ in range(NUMBER_OF_CHANNELS)],
                                     NUMBER_OF_REQUESTS, ARRIVAL_RATES[payload_size],
                                     waiting_times, service_times, comm_delays))

        environment.run()

        print('[DEBUG] ReferenceModels mUAV simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Comm Delay = {np.mean(service_times)} seconds.')

        print('[DEBUG] ReferenceModels mUAV simulate_ops: '
              f'Payload Size = {payload_size / 1e6} Mb | '
              f'Average Wait Delay = {np.mean(waiting_times)} seconds.')

        print('[INFO] ReferenceModels mUAV simulate_ops: Decode-Forward with UAV-relays | '
              f'{number_of_uavs} UAV-relays | M/G/{NUMBER_OF_CHANNELS} queuing at the data channels | '
              f'Payload Size = {payload_size / 1e6} Mb | UAV Power Consumption = {static_uav_pavg / 1e3} kW | '
              f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(waiting_times, service_times))} seconds.\n')


# Run Trigger
if __name__ == '__main__':
    simulate_ops()
