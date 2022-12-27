"""
This script encapsulates the SMDP value iteration and the projected subgradient ascent algorithms employed in MAESTRO
(our adaptive scheduling & trajectory design scheme for multi-winged, power-constrained, rotary UAVs).

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import os

"""
Configurations-I: Tensorflow logging | XLA-JIT enhancement
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ~/workspace/repos/MAESTRO-X/MAESTRO.py'

import re
import sys
import uuid
import warnings
import traceback
import numpy as np
import tensorflow as tf
from threading import Lock
from collections import namedtuple
from scipy.optimize import minimize
from dataclasses import dataclass, astuple
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# Numpy random seed | Numpy print options
np.random.seed(6)
np.set_printoptions(threshold=sys.maxsize)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# A global resource access semaphore for concurrent evaluations
lock = Lock()

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)
deg2rad, rad2deg = lambda _x: (np.pi / 180.0) * _x, lambda _x: (180.0 / np.pi) * _x

"""
Configurations-II: Global simulation parameters
"""

# The data payload sizes for this evaluation ($L$) in bits
DATA_PAYLOAD_SIZES = [1e6, 10e6, 100e6]

# The HCSO metrics in our re-formulation (new $\alpha$)
HCSO_METRICS_ALPHA = np.arange(start=0.0, stop=1.1, step=0.1)

# The max number of concurrent workers allowed in this evaluation
NUMBER_OF_WORKERS = 1024

# The output directory in which the logs from these evaluations are to be logged
OUTPUT_DIR = '../../../logs/policies/'

# The UAV average power constraints for this evaluation ($P_{\text{avg}}$) in Watts
AVG_POWER_CONSTRAINTS = np.arange(start=1e3, stop=2.2e3, step=0.2e3)

# Raw arrival rates which will be scaled w.r.t the rate factor and the load escalation factor
RAW_ARRIVAL_RATES = {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}

# Low congestion ($\Lambda'$): 1.0 Mb: 5-reqs/1-min | 10.0 Mb: 1-req/1-min | 100.0 Mb: 1-req/6-min
LOAD_ESCALATION = 1
LOW_ARRIVAL_RATES = {_k: _v * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

# High congestion ($\Lambda'$): Load escalation factor * Arrival rates for the "low congestion" regime
LOAD_ESCALATION = 100
HIGH_ARRIVAL_RATES = {_k: _v * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

# The input directory in which the logs from the associated HCSO evaluations have been logged
INPUT_DIR = {dp_size: f'../../../logs/policies/{int(dp_size / 1e6)}/' for dp_size in DATA_PAYLOAD_SIZES}

# Moderate congestion ($\Lambda'$): Load escalation factor * Arrival rates for the "moderate congestion" regime
LOAD_ESCALATION = 10
MODERATE_ARRIVAL_RATES = {_k: _v * LOAD_ESCALATION for _k, _v in RAW_ARRIVAL_RATES.items()}

"""
Utilities
"""


def log_outputs(identifier, dual_value, data_payload_size, power_const, wait_states, wait_actions,
                comm_states, comm_actions, comm_delays, energy_values, bs_delays, bs_energies, uav_delays,
                uav_energies, optimal_trajs, optimal_velos, relay_statuses, optimal_wait_policy, optimal_comm_policy):
    """
    Log outputs
    """
    pwr_const = int(power_const)
    dpl_size = int(data_payload_size / 1e6)
    file = f'{OUTPUT_DIR}{dpl_size}-{pwr_const}.log'

    tf.io.write_file(file, tf.strings.format('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}',
                                             (tf.constant(str(identifier), dtype=tf.string),
                                              tf.constant(str(dual_value), dtype=tf.string),
                                              tf.constant(str(power_const), dtype=tf.string),
                                              tf.constant(str(data_payload_size), dtype=tf.string),
                                              tf.constant(str(bs_delays.numpy()), dtype=tf.string),
                                              tf.constant(str(uav_delays.numpy()), dtype=tf.string),
                                              tf.constant(str(comm_delays.numpy()), dtype=tf.string),
                                              tf.constant(str(bs_energies.numpy()), dtype=tf.string),
                                              tf.constant(str(wait_states.numpy()), dtype=tf.string),
                                              tf.constant(str(comm_states.numpy()), dtype=tf.string),
                                              tf.constant(str(uav_energies.numpy()), dtype=tf.string),
                                              tf.constant(str(wait_actions.numpy()), dtype=tf.string),
                                              tf.constant(str(comm_actions.numpy()), dtype=tf.string),
                                              tf.constant(str(energy_values.numpy()), dtype=tf.string),
                                              tf.constant(str(optimal_trajs.numpy()), dtype=tf.string),
                                              tf.constant(str(optimal_velos.numpy()), dtype=tf.string),
                                              tf.constant(str(relay_statuses.numpy()), dtype=tf.string),
                                              tf.constant(str(optimal_wait_policy.numpy()), dtype=tf.string),
                                              tf.constant(str(optimal_comm_policy.numpy()), dtype=tf.string))))


"""
Core operations
"""


class MAESTRO(object):
    """
    MAESTRO
    """

    """
    Configurations-III: Simulation parameters
    """

    ''' Deployment model '''

    # The height of the BS from the ground ($H_{B}$) in meters
    BASE_STATION_HEIGHT = 80.0

    # The height of the UAV from the ground ($H_{U}$) in meters
    UAV_HEIGHT = 200.0

    # The propagation environment under analysis
    DEPLOYMENT_ENVIRONMENT = 'rural'  # 'rural', 'urban', 'suburban'

    # The radius of the circular cell under evaluation ($a$) in meters
    CELL_RADIUS = 1e3

    # The number of radii needed for discretization of comm state space $G_{R}+1$ or K_{R}+1$ or $N_{\text{sp}}$
    RADII_LEVELS = 25

    # The smallest required distance between two nodes (UAV/GN) along the circumference of a specific radius level in m
    MIN_CIRC_DISTANCE = 25.0

    ''' UAV mobility power consumption model '''

    # The mean rotor-induced velocity ($v_{0}$) in m/s
    MEAN_ROTOR_INDUCED_VELOCITY = 7.2

    # The maximum UAV velocity ($V_{\text{max}}$) in m/s
    MAX_UAV_VELOCITY = 55.0

    # The rotor blade tip speed ($U_{\text{tip}}$) in m/s
    ROTOR_BLADE_TIP_SPEED = 200.0

    # The UAV power consumption profile constant 1 ($P_{1}$) relevant for blade profile evaluation
    POWER_PROFILE_CONSTANT_1 = 580.65

    # The UAV power consumption profile constant 3 ($P_{3}$) which corresponds to the parasite term
    POWER_PROFILE_CONSTANT_3 = 0.0073

    # The UAV power consumption profile constant 2 ($P_{2}$) relevant for induced velocity profile evaluation
    POWER_PROFILE_CONSTANT_2 = 790.6715

    # The thrust-to-weight ratio in the rotary-wing UAV motion model ($\kappa_{\text{UAV}}{\triangleq}\frac{T}{W}$
    THRUST_TO_WEIGHT_RATIO = 1.0

    # The number of UAV radial velocity "levels" needed for discretization of the waiting action space $R_{\text{sp}}$
    VELOCITY_LEVELS = 25

    ''' Channel model '''

    # The additional NLoS attenuation factor ($\kappa$)
    NLoS_ATTENUATION_CONSTANT = 0.2

    # The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
    LoS_PATH_LOSS_EXPONENT = 2.0

    # The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
    NLoS_PATH_LOSS_EXPONENT = 2.8

    '''
    TODO: Change this number-of-data-channels parameter according to the deployment environment (Verizon LTE/LTE-A/5G)
    '''
    # The number of data channels ($N_{C}$) in this deployment
    if DEPLOYMENT_ENVIRONMENT == 'rural':
        TOTAL_BANDWIDTH = 10e6
        NUMBER_OF_CHANNELS = 2  # Verizon rural: 2x 5-MHz LTE-A
    elif DEPLOYMENT_ENVIRONMENT == 'urban':
        TOTAL_BANDWIDTH = 40e6
        NUMBER_OF_CHANNELS = 8  # Verizon NYC: 10x 5-MHz LTE-A
    else:
        TOTAL_BANDWIDTH = 20e6
        NUMBER_OF_CHANNELS = 4  # Verizon suburban: 4x 5-MHz LTE-A

    # The bandwidth available per orthogonal data channel ($B$) in Hz
    CHANNEL_BANDWIDTH = TOTAL_BANDWIDTH / NUMBER_OF_CHANNELS

    # The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$)
    REFERENCE_SNR_AT_1_METER = linear((5e6 * 40) / CHANNEL_BANDWIDTH)

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

    ''' Traffic generation model '''

    # The effective arrival rates for the deployment under analysis
    if DEPLOYMENT_ENVIRONMENT == 'rural':
        ARRIVAL_RATES = LOW_ARRIVAL_RATES
    elif DEPLOYMENT_ENVIRONMENT == 'urban':
        ARRIVAL_RATES = HIGH_ARRIVAL_RATES
    else:
        ARRIVAL_RATES = MODERATE_ARRIVAL_RATES

    # A waiting state interval: a time period in which no additional request is received ($\Delta_{0}$) in seconds
    WAITING_STATE_INTERVAL = {dp_size: -np.log(0.93) / ARRIVAL_RATES[dp_size] for dp_size in DATA_PAYLOAD_SIZES}

    ''' Algorithmic model '''

    # The maximum number of trajectory segments allowed in the HCSO solution ($M_{\text{max}}$)
    MAX_TRAJECTORY_SEGMENTS = 32

    # The termination threshold for the SMDP Value Iteration (VITER) algorithm ($\delta$), i.e.,
    #   terminate if $\max_{s{\in}\mathcal{S}} H(s) - \min(x_{s{\in}\mathcal{S}} H(s)) < \delta$
    VITER_TERMINATION_THRESHOLD = 1e-3

    # The initial dual variable step-size in the projected sub-gradient ascent algorithm ($\rho_{0}$)
    INITIAL_DUAL_VARIABLE_STEP_SIZE = 0.1

    # The primal feasibility threshold in the projected sub-gradient ascent algorithm ($\epsilon_{PF}$)
    PRIMAL_FEASIBILITY_THRESHOLD = 1e-3

    # The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
    BISECTION_METHOD_TOLERANCE = 1e-10

    # The complementary slackness threshold in the projected sub-gradient ascent algorithm ($\epsilon_{CS}$)
    COMPLEMENTARY_SLACKNESS_THRESHOLD = 0.1

    # The dual variable convergence threshold in the projected sub-gradient ascent algorithm ($\epsilon_{DI}$)
    DUAL_CONVERGENCE_THRESHOLD = 1e-3

    # A namedtuple constituting all the relevant penalty metrics involved in the CSO/HCSO cost function evaluation
    PENALTIES_CAPSULE = namedtuple('penalties_capsule', ['t_p_1', 't_p_2', 'e_p_1', 'e_p_2'])

    # The convergence confidence level for the bisection method to find the optimal value of $Z$ for rate adaptation
    BISECTION_CONVERGENCE_CONFIDENCE = 10

    # The convergence confidence level for the SMDP-VI and PSGA algorithms to find the optimal MAESTRO control policy
    CONVERGENCE_CONFIDENCE = 10

    # The number of waypoints to be interpolated between any two given points in the generated $M$-segment trajectories
    INTERPOLATION_FACTOR = 2

    # The termination threshold for angular velocity optimization (within Lagrangian minimization) in the waiting states
    ANGULAR_VELOCITY_TERMINATION_THRESHOLD = 1e-10

    """
    DTOs
    """

    @dataclass(order=True)
    class SMDPValueIterationDataCapsule:
        nu: np.float64  # The dual variable ($\nu$)
        o_star: tf.Variable  # The optimal wait policy (O^{*})
        u_star: tf.Variable  # The optimal comm policy (U^{*})
        u_star_indices: tf.Variable  # The optimal comm action indices
        g: np.float64  # The dual function ($g$) obtained via the primal-Lagr formulation
        e_bar: np.float64  # The average UAV energy consumption for a canary state ($\Bar{E}$)
        t_bar: np.float64  # The average time spent by the UAV per state w.r.t a canary state ($\Bar{T}$)

    """
    Algorithms
    """

    def __init__(self, id__, average_power_constraint, payload_size, num_workers, evaluators__):
        self.id = id__
        self.dual_var = 0.0
        self.num_workers = num_workers
        self.payload_size = payload_size  # The data payload size for this "run"
        self.average_power_constraint = average_power_constraint  # The average power constraint for this "run"

        self.ARRIVAL_RATE = self.ARRIVAL_RATES[self.payload_size]
        self.WAITING_STATE_INTERVAL = self.WAITING_STATE_INTERVAL[self.payload_size]

        print(f'[INFO] [{self.id}] MAESTRO Initialization: Bringing things up - '
              f'L = {self.payload_size / 1e6} Mb | P_avg = {self.average_power_constraint / 1e3} kW.')

        radii = np.linspace(start=0.0, stop=self.CELL_RADIUS, num=self.RADII_LEVELS)
        vels = np.linspace(start=-self.MAX_UAV_VELOCITY, stop=self.MAX_UAV_VELOCITY, num=self.VELOCITY_LEVELS)

        self.waiting_states = tf.constant(radii, dtype=tf.float64)
        self.waiting_actions = tf.constant(vels, dtype=tf.float64)

        angles = [np.linspace(start=0.0, stop=2 * np.pi, num=int((2 * np.pi * _r) /
                                                                 self.MIN_CIRC_DISTANCE) + 1) for _r in radii]

        coords_dict = {_r: _r * np.einsum('ji', np.vstack([np.cos(angles[_i]),
                                                           np.sin(angles[_i])])) for _i, _r in enumerate(radii)}

        comm_states_dict = {_r_u: [{_r_g: np.unique(np.rad2deg([
            (np.arctan2(*__c_u[::-1]) - np.arctan2(*__c_g[::-1])) % (2 * np.pi) for __c_g in _c_g]))
            for _r_g, _c_g in coords_dict.items()} for __c_u in _c_u] for _r_u, _c_u in coords_dict.items()}

        comm_states_arr = []
        for _r_u, _csd_u in comm_states_dict.items():
            for _csd_gu in _csd_u:
                for _r_g, _csd_g in _csd_gu.items():
                    for _a_gu in _csd_g:
                        comm_states_arr.append([int(_r_u), int(_r_g), int(_a_gu)])

        self.comm_actions = tf.constant(radii, dtype=tf.float64)
        self.comm_states = tf.constant(np.unique(comm_states_arr, axis=0), dtype=tf.float64)

        self.o_star, self.u_star, self.u_star_indices = None, None, None
        self.relay_status = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.int8)

        self.bs_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.uav_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.comm_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)

        self.energy_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.bs_nrg_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.uav_nrg_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)

        self.optimal_velocities = tf.Variable(tf.zeros(
            shape=[self.comm_states.shape[0], self.comm_actions.shape[0],
                   self.INTERPOLATION_FACTOR * self.MAX_TRAJECTORY_SEGMENTS], dtype=tf.float64), dtype=tf.float64)

        self.optimal_trajectories = tf.Variable(tf.zeros(
            shape=[self.comm_states.shape[0], self.comm_actions.shape[0],
                   self.INTERPOLATION_FACTOR * self.MAX_TRAJECTORY_SEGMENTS, 2], dtype=tf.float64), dtype=tf.float64)

        evaluators__.append(self)  # Self-Registration

    def __enter__(self):
        return self

    def __evaluate_power_consumption(self, uav_flying_velocity):
        v, u_tip, v_0 = uav_flying_velocity, self.ROTOR_BLADE_TIP_SPEED, self.MEAN_ROTOR_INDUCED_VELOCITY
        p_1, p_2, p_3 = self.POWER_PROFILE_CONSTANT_1, self.POWER_PROFILE_CONSTANT_2, self.POWER_PROFILE_CONSTANT_3

        return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_3 * (v ** 3)) + \
            (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)

    def __f_z(self, z):
        b = self.CHANNEL_BANDWIDTH
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
        assert tolerance is not None
        mid, converged, conf, conf_th = 0.0, False, 0, self.BISECTION_CONVERGENCE_CONFIDENCE

        while not converged or conf < conf_th:
            mid = (high + low) / 2
            if f(low, *args) * f(high, *args) > 0.0:
                low = mid
            else:
                high = mid
            converged = abs(high - low) < tolerance
            conf += 1 if converged else -conf

        return mid

    def __z(self, gamma):
        b = self.CHANNEL_BANDWIDTH
        return np.sqrt(2 * ((2 ** (gamma / b)) - 1))

    def __u(self, gamma, d, los):
        alpha = self.LoS_PATH_LOSS_EXPONENT
        alpha_ = self.NLoS_PATH_LOSS_EXPONENT
        kappa = self.NLoS_ATTENUATION_CONSTANT
        b, gamma_ = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER
        return ((2 ** (gamma / b)) - 1) / (gamma_ * 1 if los else kappa * (d ** -alpha if los else -alpha_))

    def __evaluate_los_throughput(self, d, phi, r_star_los):
        k_1, k_2 = self.LoS_RICIAN_FACTOR_1, self.LoS_RICIAN_FACTOR_2,
        k, alpha = k_1 * np.exp(k_2 * phi), self.LoS_PATH_LOSS_EXPONENT
        b, gamma_ = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER
        df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), self.BISECTION_METHOD_TOLERANCE

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ * (d ** -alpha))), t)

        gamma_star = self.__f_z(z_star)
        tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)

    def __evaluate_nlos_throughput(self, d, r_star_nlos):
        b, gamma_ = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER
        alpha_, kappa = self.NLoS_PATH_LOSS_EXPONENT, self.NLoS_ATTENUATION_CONSTANT
        df, nc, y, t = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_))), self.BISECTION_METHOD_TOLERANCE

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(b * np.log2(1 + (
                                       rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa * (d ** -alpha_))), t)

        gamma_star = self.__f_z(z_star)
        tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)

    def __calculate_adapted_throughput(self, d, phi, r_star_los, r_star_nlos):
        n_w = self.num_workers
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__evaluate_nlos_throughput, d, r_star_nlos)
            executor.submit(self.__evaluate_los_throughput, d, phi, r_star_los)

    # noinspection PyMethodMayBeStatic
    def __interpolate_waypoints(self, p_indices, p, res_multiplier):
        m = len(p_indices)
        spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)

        return tf.constant(list(zip(spl_x(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)),
                                    spl_y(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)))))

    # noinspection PyMethodMayBeStatic
    def __interpolate_velocities(self, v_indices, v, res_multiplier):
        m, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
        return spl_v(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64))

    def __penalties(self, p__, v__, x_g, res_multiplier):
        n_w = self.num_workers
        min_power = self.__evaluate_power_consumption(22.0)
        z_1, z_2 = self.PROPAGATION_ENVIRONMENT_PARAMETER_1, self.PROPAGATION_ENVIRONMENT_PARAMETER_2

        p = self.__interpolate_waypoints([_ for _ in range(p__.shape[0])], p__, res_multiplier)

        midpoint, h_uav, h_bs = int(p.shape[0] / 2), self.UAV_HEIGHT, self.BASE_STATION_HEIGHT

        v = self.__interpolate_velocities([_ for _ in range(v__.shape[0])], v__, res_multiplier)

        t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1),
                      tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

        # Decode (GN --> UAV)

        r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
        h_u = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)

        d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
        phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()

        r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
                executor.submit(self.__calculate_adapted_throughput, d_gu__, phi_gu__, r_los_gu[i__], r_nlos_gu[i__])

        phi_degrees_gu = (180.0 / np.pi) * phi_gu

        p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
        p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)

        r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

        h_1 = self.payload_size - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))

        t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
        e_p_1 = (lambda: 0.0, lambda: (min_power * t_p_1))[h_1.numpy() > 0.0]()

        # Forward (UAV --> BS)

        r_ub = tf.norm(p[midpoint:], axis=1)
        h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)

        d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
        phi_ub = tf.asin(tf.divide(h_ub, d_ub))

        r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
                executor.submit(self.__calculate_adapted_throughput, d_ub__, phi_ub__, r_los_ub[i__], r_nlos_ub[i__])

        phi_degrees_ub = (180.0 / np.pi) * phi_ub

        p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
        p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)

        r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))

        h_2 = self.payload_size - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))

        t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
        e_p_2 = (lambda: 0.0, lambda: (min_power * t_p_2))[h_2.numpy() > 0.0]()

        return self.PENALTIES_CAPSULE(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)

    @tf.function(experimental_relax_shapes=True)
    @tf.autograph.experimental.do_not_convert
    def __power_cost(self, v):
        n_w = self.num_workers
        return tf.map_fn(self.__evaluate_power_consumption, v, parallel_iterations=n_w)

    def __calculate_comm_cost(self, p, v, nu, x_g, f_hat, e_hat=None, t_hat=None):
        p_average, interp = self.average_power_constraint, self.INTERPOLATION_FACTOR

        t_p_1, t_p_2, e_p_1, e_p_2 = self.__penalties(p, v, x_g, interp)

        t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1),
                      tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

        t__ = tf.reduce_sum(t) + t_p_1 + t_p_2
        e__ = tf.reduce_sum(tf.multiply(t, self.__power_cost(v[:-1]))) + e_p_1 + e_p_2

        tf.compat.v1.assign(f_hat, ((1.0 - nu * p_average) * t__ + nu * e__), validate_shape=True, use_locking=True)

        if e_hat is not None:
            tf.compat.v1.assign(e_hat, e__, validate_shape=True, use_locking=True)

        if t_hat is not None:
            tf.compat.v1.assign(t_hat, t__, validate_shape=True, use_locking=True)

    # noinspection PyUnusedLocal
    def __read_optimized_trajectories(self, state_index, action_index,
                                      c_state, c_action, dual_variable, lagrangian, energy, time_duration):
        nu, f_hat = dual_variable, tf.Variable(0.0, dtype=tf.float64)
        i__, j__, a_num = state_index, action_index, self.comm_actions.shape[0]
        n_w, p_size, h_b = self.num_workers, self.payload_size, self.BASE_STATION_HEIGHT
        z_1, z_2 = self.PROPAGATION_ENVIRONMENT_PARAMETER_1, self.PROPAGATION_ENVIRONMENT_PARAMETER_2

        opt_traj, opt_velo = self.optimal_trajectories, self.optimal_velocities
        l_comm_star, e_comm_star, t_comm_star = lagrangian, energy, time_duration
        xi_s, delays, nrgs = self.relay_status, self.comm_delays, self.energy_vals
        e_usage, delta = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
        bs_delays, bs_nrgs, uav_delays, uav_nrgs = self.bs_delays, self.bs_nrg_vals, self.uav_delays, self.uav_nrg_vals

        # Read the trajectories (wps and vels) from the decoupled HCSO runs (for different h_alphas)

        read_trajs = []
        traj_files = [f'{INPUT_DIR[p_size]}{_h_a}/trajs/{(i__ * a_num) + j__ + 1}.log' for _h_a in HCSO_METRICS_ALPHA]

        for traj_file in traj_files:
            args = []

            with open(traj_file, 'r') as file:
                for line in file.readlines():
                    # noinspection RegExpUnnecessaryNonCapturingGroup
                    args.append(tf.string.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)', line.strip()), tf.float64))

            file_v_star, file_p_star = args[2:]
            read_trajs.append((file_p_star, file_v_star))

        r_u, r_g, psi_gu = c_state
        num_trajs = len(read_trajs)
        f_hats = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
        deltas = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
        e_usages = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
        x_g = tf.constant([r_g * np.cos(psi_gu), r_g * np.sin(psi_gu)], dtype=tf.float64)

        # Pick the best trajectory (out of all h_alpha variations) as the one that minimizes the cost metric

        for traj_idx in range(num_trajs):
            with ThreadPoolExecutor(max_workers=n_w) as executor:
                p_star_, v_star_ = read_trajs[traj_idx]
                executor.submit(self.__calculate_comm_cost, p_star_, v_star_,
                                nu, x_g, f_hats[traj_idx], e_usages[traj_idx], deltas[traj_idx])

        min_traj_idx = tf.argmin(f_hats, axis=0)
        p_star, v_star = read_trajs[min_traj_idx]

        # Lagrangian cost determination for scheduling (Direct BS or UAV relay?)

        d_gb = np.sqrt(np.add(np.square(h_b), np.square(np.linalg.norm(x_g))))
        phi_gb = np.arcsin(np.divide(h_b, d_gb))

        r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__calculate_adapted_throughput, d_gb, phi_gb, r_los_gb, r_nlos_gb)
            executor.submit(self.__calculate_comm_cost, p_star, v_star, nu, x_g, f_hat, e_usage, delta)

        phi_degrees_gb = (180.0 / np.pi) * phi_gb

        p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
        p_nlos_gb = 1 - p_los_gb

        r_bar_gb = tf.add(tf.multiply(p_los_gb, r_los_gb), tf.multiply(p_nlos_gb, r_nlos_gb))

        l_xi_0 = (p_size / r_bar_gb) if r_bar_gb != 0.0 else np.inf
        l_xi_1, e_xi_1, t_xi_1 = f_hat.numpy(), e_usage.numpy(), delta.numpy()
        l__, e__, t__, xi__ = (l_xi_1, e_xi_1, t_xi_1, 1) if (l_xi_1 < l_xi_0) else (l_xi_0, 0.0, l_xi_0, 0)

        # Tensor updates for further processing
        tf.compat.v1.assign(l_comm_star[i__, j__], l__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(e_comm_star[i__, j__], e__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(t_comm_star[i__, j__], t__, validate_shape=True, use_locking=True)

        # Class-wide data collection updates for evaluation
        tf.compat.v1.assign(nrgs[i__], e__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(xi_s[i__], xi__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(delays[i__], t__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(bs_nrgs[i__], 0.0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(uav_nrgs[i__], e_xi_1, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(bs_delays[i__], l_xi_0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(uav_delays[i__], t_xi_1, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(opt_traj[i__, j__], p_star, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(opt_velo[i__, j__], v_star, validate_shape=True, use_locking=True)

    def __angular_velocity_optimization(self, state_index, action_index, uav_position,
                                        radial_velocity, dual_variable, lagrangians, energies, time_durations):
        l_wait_star, e_wait_star, t_wait_star = lagrangians, energies, time_durations
        theta_c, delta_c = np.random.random(), self.ANGULAR_VELOCITY_TERMINATION_THRESHOLD
        id__, p_avg, delta_0 = self.id, self.average_power_constraint, self.WAITING_STATE_INTERVAL
        i__, j__, r_u, v_r, nu = state_index, action_index, uav_position.numpy(), radial_velocity.numpy(), dual_variable

        v_max, flying_velocity = self.MAX_UAV_VELOCITY, lambda x: ((v_r ** 2) + ((r_u ** 2) * (x ** 2))) ** 0.5

        def objective(x):
            return nu * (self.__evaluate_power_consumption(flying_velocity(x)) - p_avg) * delta_0

        constraints = ({'type': 'ineq', 'fun': lambda x: v_max - flying_velocity(x)})
        theta_c_star = minimize(objective, theta_c, method='SLSQP', constraints=constraints, tol=delta_c).x[0]

        p_mob_star = self.__evaluate_power_consumption(flying_velocity(theta_c_star))

        tf.compat.v1.assign(l_wait_star[i__, j__],
                            nu * (p_mob_star - p_avg) * delta_0,
                            validate_shape=True, use_locking=True)
        tf.compat.v1.assign(t_wait_star[i__, j__], delta_0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(e_wait_star[i__, j__], p_mob_star * delta_0, validate_shape=True, use_locking=True)

    def __optimize_waiting_states(self, nu, s_wait, a_wait, l_wait_star, e_wait_star, t_wait_star):
        n_w = self.num_workers

        try:

            with ThreadPoolExecutor(max_workers=n_w) as executor:
                for i__, r_u in enumerate(s_wait):
                    for j__, v_r in enumerate(a_wait):
                        executor.submit(self.__angular_velocity_optimization,
                                        i__, j__, r_u, v_r, nu, l_wait_star, e_wait_star, t_wait_star)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO __optimize_waiting_states: Exception caught '
                  f'during waiting state optimization - {traceback.print_tb(e__.__traceback__)}.')

    def __optimize_comm_states(self, nu, s_comm, a_comm, l_comm_star, e_comm_star, t_comm_star):
        n_w = self.num_workers

        try:

            with ThreadPoolExecutor(max_workers=n_w) as executor:
                for i__, s in enumerate(s_comm):
                    for j__, r_u in enumerate(a_comm):
                        executor.submit(self.__read_optimized_trajectories,
                                        i__, j__, s, r_u, nu, l_comm_star, e_comm_star, t_comm_star)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO __optimize_comm_states: Exception caught '
                  f'during comm state optimization - {traceback.print_tb(e__.__traceback__)}.')

    def __smdp_waiting_viter_updates(self, l_wait_star, e_wait_star, t_wait_star, v_i_wait, v_i_comm,
                                     h_i_wait, e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait, o_star):
        try:

            lambda__, delta_0 = self.ARRIVAL_RATE, self.WAITING_STATE_INTERVAL

            v_i_1_wait = tf.constant(v_i_wait)

            s_wait_added = tf.add(tf.tile(tf.expand_dims(s_wait, axis=1),
                                          multiples=[1, a_wait.shape[0]]), delta_0 * a_wait)

            wait_indices = tf.argmin(tf.abs(tf.subtract(
                tf.tile(tf.expand_dims(s_wait_added, axis=2), multiples=[1, 1, s_wait.shape[0]]), s_wait)), axis=2)

            # Waiting state value functions: Transition to another waiting state
            v_i_wait_remapped = tf.gather(tf.tile(tf.expand_dims(v_i_wait, axis=1),
                                                  multiples=[1, a_wait.shape[0]]), wait_indices)[:, :, 0]

            # Comm state value functions: Transition to a comm state

            mins_org = tf.reshape(tf.gather(s_wait, wait_indices), shape=[-1])

            v_i_comm_remapped = tf.reshape(
                tf.reduce_sum(tf.map_fn(lambda x: tf.gather(v_i_comm,
                                                            tf.where(tf.equal(s_comm[:, 0], x))),
                                        mins_org), axis=1), shape=[s_wait.shape[0], a_wait.shape[0]])

            _v_i_added = tf.add(l_wait_star,
                                np.exp(-lambda__ * delta_0) * v_i_wait_remapped,
                                (1 - np.exp(-lambda__ * delta_0)) * v_i_comm_remapped)

            o_i_1 = tf.argmin(_v_i_added, axis=1)
            o_star__ = tf.gather(a_wait, o_i_1)

            tf.compat.v1.assign(v_i_wait, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(h_i_wait, tf.subtract(v_i_wait, v_i_1_wait), validate_shape=True, use_locking=True)

            tf.compat.v1.assign(o_star, o_star__, validate_shape=True, use_locking=True)

            s_wait_added_min = tf.add(s_wait, delta_0 * o_star__)
            wait_indices_min = tf.squeeze(tf.argmin(tf.abs(tf.subtract(
                tf.tile(tf.expand_dims(s_wait_added_min, axis=1), multiples=[1, s_wait.shape[0]]), s_wait)), axis=1))

            # Waiting state energy & time costs

            e_wait_star_remapped = tf.boolean_mask(e_wait_star, tf.one_hot(
                o_i_1, e_wait_star.shape[1], on_value=True, off_value=False, dtype=tf.bool))

            e_i_wait_remapped = tf.gather(e_i_wait, wait_indices_min)

            t_wait_star_remapped = tf.boolean_mask(t_wait_star, tf.one_hot(
                o_i_1, t_wait_star.shape[1], on_value=True, off_value=False, dtype=tf.bool))

            t_i_wait_remapped = tf.gather(t_i_wait, wait_indices_min)

            # Comm state energy & time costs

            mins_org_min = tf.gather(s_wait, wait_indices_min)

            e_i_comm_remapped = tf.squeeze(tf.reduce_sum(
                tf.map_fn(lambda x: tf.gather(e_i_comm, tf.where(tf.equal(s_comm[:, 0], x))), mins_org_min), axis=1))

            t_i_comm_remapped = tf.squeeze(tf.reduce_sum(
                tf.map_fn(lambda x: tf.gather(t_i_comm, tf.where(tf.equal(s_comm[:, 0], x))), mins_org_min), axis=1))

            _e_i_added = tf.add(e_wait_star_remapped,
                                np.exp(-lambda__ * delta_0) * e_i_wait_remapped,
                                (1 - np.exp(-lambda__ * delta_0)) * e_i_comm_remapped)

            tf.compat.v1.assign(e_i_wait, _e_i_added, validate_shape=True, use_locking=True)

            _t_i_added = tf.add(t_wait_star_remapped,
                                np.exp(-lambda__ * delta_0) * t_i_wait_remapped,
                                (1 - np.exp(-lambda__ * delta_0)) * t_i_comm_remapped)

            tf.compat.v1.assign(t_i_wait, _t_i_added, validate_shape=True, use_locking=True)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO __smdp_waiting_viter_updates: Exception caught '
                  f'during SMDP VITER waiting state updates - {traceback.print_tb(e__.__traceback__)}.')

    def __smdp_comm_viter_updates(self, l_comm_star, e_comm_star, t_comm_star, v_i_comm, v_i_wait, h_i_comm,
                                  e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm, u_star, u_star_indices):
        try:

            v_i_1_comm = tf.constant(v_i_comm)

            _v_i_wait = tf.tile(tf.expand_dims(
                tf.squeeze(tf.map_fn(lambda x: tf.gather(v_i_wait,
                                                         tf.where(tf.equal(s_wait, x))),
                                     a_comm)), axis=0), multiples=[l_comm_star.shape[0], 1])

            _v_i_added = tf.add(l_comm_star, _v_i_wait)  # Transition only to the waiting state

            u_i_1 = tf.argmin(_v_i_added, axis=1)
            u_star__ = tf.gather(a_comm, u_i_1, axis=0)

            tf.compat.v1.assign(u_star, u_star__, validate_shape=True, use_locking=True)
            tf.compat.v1.assign(u_star_indices, u_i_1, validate_shape=True, use_locking=True)
            tf.compat.v1.assign(v_i_comm, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(h_i_comm, tf.subtract(v_i_comm, v_i_1_comm), validate_shape=True, use_locking=True)

            # Comm state energy costs

            e_i_wait_remapped = tf.squeeze(tf.map_fn(
                lambda x: tf.gather(e_i_wait, tf.where(tf.equal(s_wait, x))), tf.norm(u_star__, axis=1)))

            e_comm_star_remapped = tf.boolean_mask(
                e_comm_star, tf.one_hot(u_i_1, e_comm_star.shape[1], on_value=True, off_value=False, dtype=tf.bool))

            tf.compat.v1.assign(e_i_comm,
                                tf.add(e_comm_star_remapped, e_i_wait_remapped), validate_shape=True, use_locking=True)

            # Comm state time duration costs

            t_i_wait_remapped = tf.squeeze(tf.map_fn(
                lambda x: tf.gather(t_i_wait, tf.where(tf.equal(s_wait, x))), tf.norm(u_star__, axis=1)))

            t_comm_star_remapped = tf.boolean_mask(
                t_comm_star, tf.one_hot(u_i_1, t_comm_star.shape[1], on_value=True, off_value=False, dtype=tf.bool))

            tf.compat.v1.assign(t_i_comm,
                                tf.add(t_comm_star_remapped, t_i_wait_remapped), validate_shape=True, use_locking=True)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO __smdp_comm_viter_updates: Exception caught '
                  f'during SMDP VITER comm state updates - {traceback.print_tb(e__.__traceback__)}.')

    def __pi_comm(self):
        lambda_, delta_0 = self.ARRIVAL_RATE, self.WAITING_STATE_INTERVAL
        p_e = np.exp(-lambda_ * delta_0)
        return (1 - p_e) / (2 - p_e)

    def __value_iteration(self, dual_variable):
        n_w = self.num_workers
        i, nu, delta = 0, dual_variable, self.VITER_TERMINATION_THRESHOLD

        try:

            # Waiting states

            s_wait, a_wait = self.waiting_states, self.waiting_actions
            s_wait_size, a_wait_size = s_wait.shape[0], a_wait.shape[0]
            l_wait_star = tf.Variable(tf.zeros(shape=[s_wait_size, a_wait_size], dtype=tf.float64), dtype=tf.float64)
            e_wait_star = tf.Variable(tf.zeros(shape=[s_wait_size, a_wait_size], dtype=tf.float64), dtype=tf.float64)
            t_wait_star = tf.Variable(tf.zeros(shape=[s_wait_size, a_wait_size], dtype=tf.float64), dtype=tf.float64)

            v_i_wait = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)
            h_i_wait = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)
            e_i_wait = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)
            t_i_wait = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)

            o_star = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)

            # Comm states

            s_comm, a_comm = self.comm_states, self.comm_actions
            s_comm_size, a_comm_size, canary_index = s_comm.shape[0], a_comm.shape[0], s_comm.shape[0] - 1

            l_comm_star = tf.Variable(tf.zeros(shape=[s_comm_size, a_comm_size], dtype=tf.float64), dtype=tf.float64)
            e_comm_star = tf.Variable(tf.zeros(shape=[s_comm_size, a_comm_size], dtype=tf.float64), dtype=tf.float64)
            t_comm_star = tf.Variable(tf.zeros(shape=[s_comm_size, a_comm_size], dtype=tf.float64), dtype=tf.float64)

            v_i_comm = tf.Variable(tf.zeros(shape=[s_comm_size, ], dtype=tf.float64), dtype=tf.float64)
            h_i_comm = tf.Variable(tf.zeros(shape=[s_comm_size, ], dtype=tf.float64), dtype=tf.float64)
            e_i_comm = tf.Variable(tf.zeros(shape=[s_comm_size, ], dtype=tf.float64), dtype=tf.float64)
            t_i_comm = tf.Variable(tf.zeros(shape=[s_comm_size, ], dtype=tf.float64), dtype=tf.float64)

            u_star = tf.Variable(tf.zeros(shape=[s_comm_size, 2], dtype=tf.float64), dtype=tf.float64)
            u_star_indices = tf.Variable(tf.zeros(shape=[s_comm_size, ], dtype=tf.int64), dtype=tf.int64)

            # Optimization (lagrangian cost [via optimal action] determination per state)

            with ThreadPoolExecutor(max_workers=n_w) as executor:

                executor.submit(self.__optimize_waiting_states,
                                nu, s_wait, a_wait, l_wait_star, e_wait_star, t_wait_star)

                executor.submit(self.__optimize_comm_states,
                                nu, s_comm, a_comm, l_comm_star, e_comm_star, t_comm_star)

            # Value function updates (via SMDP state transitions: one step future look-ahead)

            converged, conf, conf_th = False, 0, self.CONVERGENCE_CONFIDENCE

            while not converged or conf < conf_th:
                with ThreadPoolExecutor(max_workers=n_w) as executor:
                    executor.submit(self.__smdp_waiting_viter_updates, l_wait_star,
                                    e_wait_star, t_wait_star, v_i_wait, v_i_comm, h_i_wait,
                                    e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait, o_star)

                    executor.submit(self.__smdp_comm_viter_updates, l_comm_star,
                                    e_comm_star, t_comm_star, v_i_comm, v_i_wait, h_i_comm,
                                    e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm, u_star, u_star_indices)

                converged = (tf.reduce_max(h_i_wait) - tf.reduce_min(h_i_wait) < delta) and \
                            (tf.reduce_max(h_i_comm) - tf.reduce_min(h_i_comm) < delta)

                conf += 1 if converged else -conf
                i += 1

            pi_comm = self.__pi_comm()
            divisor = 1 / (pi_comm * i) if pi_comm != 0.0 else np.inf

            g = divisor * v_i_comm[canary_index].numpy()
            e_bar = divisor * e_i_comm[canary_index].numpy()
            t_bar = divisor * t_i_comm[canary_index].numpy()

            return self.SMDPValueIterationDataCapsule(nu=nu, o_star=o_star, u_star=u_star,
                                                      u_star_indices=u_star_indices, g=g, e_bar=e_bar, t_bar=t_bar)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO __value_iteration: Exception caught '
                  f'during value iteration updates - {traceback.print_tb(e__.__traceback__)}.')

        return None

    def projected_subgradient_ascent(self):
        th_d, th_pf = self.DUAL_CONVERGENCE_THRESHOLD, self.PRIMAL_FEASIBILITY_THRESHOLD
        k, g_k_1, p_av, rho_0 = 0, 0.0, self.average_power_constraint, self.INITIAL_DUAL_VARIABLE_STEP_SIZE
        th_cs, converged, conf, conf_th = self.COMPLEMENTARY_SLACKNESS_THRESHOLD, False, 0, self.CONVERGENCE_CONFIDENCE

        try:

            nu_k, o_star_k, u_star_k, u_star_indices_k, g_k, e_k, t_k = astuple(self.__value_iteration(0.01))

            while not converged or conf < conf_th:
                g_k_1, nu_k = g_k, max(nu_k + ((rho_0 / (k + 1)) * (e_k - (p_av * t_k))), 0)
                nu_k, o_star_k, u_star_k, u_star_indices_k, g_k, e_k, t_k = astuple(self.__value_iteration(nu_k))

                converged = (abs(g_k - g_k_1) < th_d) and \
                            (e_k - (p_av * t_k) < th_pf) and (nu_k * abs(e_k - (p_av * t_k)) < th_cs)

                conf += 1 if converged else -conf
                k += 1

            self.dual_var, self.o_star, self.u_star, self.u_star_indices = nu_k, o_star_k, u_star_k, u_star_indices_k

            # noinspection PyTypeChecker
            log_outputs(self.id, self.dual_var, self.payload_size, self.average_power_constraint, self.waiting_states,
                        self.waiting_actions, self.comm_states, self.comm_actions, self.comm_delays,
                        self.energy_vals, self.bs_delays, self.bs_nrg_vals, self.uav_delays,
                        self.uav_nrg_vals, self.optimal_trajectories,
                        self.optimal_velocities, self.relay_status,
                        self.o_star, self.u_star)

        except Exception as e__:
            print(f'[ERROR] [{self.id}] MAESTRO projected_subgradient_ascent: Exception caught '
                  f'during projected sub-gradient ascent updates - {traceback.print_tb(e__.__traceback__)}.')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] [{self.id}] MAESTRO Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


def launch_evaluation(id_, power_constraint, data_pl_size, evaluators_):
    with MAESTRO(id_, power_constraint, data_pl_size, NUMBER_OF_WORKERS, evaluators_) as evaluator__:
        evaluator__.projected_subgradient_ascent()


# Run Trigger
if __name__ == '__main__':
    print('[INFO] [Main Thread] MAESTRO main: Starting the evaluation of the proposed SMDP formulation '
          'for adaptive multi-scale scheduling and trajectory optimization of power-constrained UAV relays...')

    evaluators = list()

    with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as exxeggutor:
        for pl_size in DATA_PAYLOAD_SIZES:
            for avg_power in AVG_POWER_CONSTRAINTS:
                exxeggutor.submit(launch_evaluation, uuid.uuid4(), avg_power, pl_size, evaluators)

    print('[INFO] [Main Thread] MAESTRO main: Completed the evaluation of the proposed SMDP formulation '
          'for adaptive multi-scale scheduling and trajectory optimization of power-constrained UAV relays for '
          f'Data Payload Sizes = {DATA_PAYLOAD_SIZES} and UAV Average Power Constraints = {AVG_POWER_CONSTRAINTS}.')
