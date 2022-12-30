"""
This script constitutes a Tensor implementation of the Hierarchical Competitive Swarm Optimization (HCSO) algorithm for
UAV path-planning within our MAESTRO framework, i.e., the two-stage SMDP-HCSO formulation of our "Adaptive UAV Active
Communication Request Scheduling and Trajectory Design for Power-Constrained Rotary-Wing UAV Relays."

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

import sys
import warnings
import traceback
import numpy as np
import tensorflow as tf
from collections import namedtuple
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

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Simulation parameters
"""

''' Deployment model '''

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The propagation environment under analysis
DEPLOYMENT_ENVIRONMENT = 'urban'  # 'rural', 'urban', 'suburban'

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1e3

# The smallest required distance between two nodes (UAV/GN) along the circumference of a specific radius level in m
# MIN_CIRC_DISTANCE = 25.0

# The number of radii "levels" needed for discretization of comm state space $G_{R}+1$ or K_{R}+1$ or $N_{\text{sp}}$
# RADII_LEVELS = 25

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

''' Channel model '''

# The additional NLoS attenuation factor ($\kappa$)
NLoS_ATTENUATION_CONSTANT = 0.2

# The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
LoS_PATH_LOSS_EXPONENT = 2.0

# The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
NLoS_PATH_LOSS_EXPONENT = 2.8

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

# The HCSO metric in our re-formulation (new $\alpha$)
HCSO_METRIC_ALPHA = 0.75  # 0.0, 0.1, 0.2, ..., 1.0

# The data payload size for this evaluation ($L$) in bits
DATA_PAYLOAD_SIZE = 10e6  # 1e6, 10e6, 100e6

# The max number of concurrent workers allowed in this evaluation
NUMBER_OF_WORKERS = 1024

# The UAV velocity and CSO/HCSO particle velocity discretization levels
CSO_VELOCITY_DISCRETIZATION_LEVELS = 25

# The initial number of trajectory segments in the HCSO algorithm ($M$)
INITIAL_TRAJECTORY_SEGMENTS = 6

# The smallest UAV velocity and CSO/HCSO particle velocity value ($V_{\text{low}}$)
CSO_MINIMUM_VELOCITY_VALUE = 0.0

# The maximum number of trajectory segments allowed in the HCSO solution ($M_{\text{max}}$)
MAXIMUM_TRAJECTORY_SEGMENTS = 32

# The output directory in which the logs from this evaluation have to be logged
OUTPUT_DIR = f'../../../logs/policies/{int(DATA_PAYLOAD_SIZE / 1e6)}_highres/{HCSO_METRIC_ALPHA}/trajs/'

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-3

# The number of initial trajectory and UAV velocity particles in the HCSO algorithm ($N$) [Swarm Size]
INITIAL_NUMBER_OF_PARTICLES = 32

# The maximum number of cost function evaluations recommended in the CSO algorithm ($\math{N}_{\text{max}}$)
MAXIMUM_COST_EVALUATIONS = 16

# A validation multiplier to make sure that the HCSO trajectory parameters are valid vis-à-vis the algorithm
HCSO_VALIDATION_MULTIPLIER = 5

# The scaling factor that determines the degree of influence of the global means in the CSO algorithm ($\omega$)
CSO_PARTICLE_VELOCITY_SCALING_FACTOR = 1.0

# The convergence confidence level for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_CONVERGENCE_CONFIDENCE = 1

# The scaling factor employed in the "disturbance around a trajectory reference solution" aspect of HCSO ($\zeta$)
HCSO_TRAJECTORY_SCALING_FACTOR = 1.0

# The scaling factor employed in the "disturbance around a velocity reference solution" aspect of HCSO ($\epsilon$)
HCSO_VELOCITY_SCALING_FACTOR = 1.0

# The number of waypoints to be interpolated between any two given points in the generated $M$-segment trajectories
INTERPOLATION_FACTOR = 2

"""
Node Deployments
"""

'''
radii = np.linspace(start=0.0, stop=CELL_RADIUS, num=RADII_LEVELS)
angles = [np.linspace(start=0.0, stop=2 * np.pi, num=int((2 * np.pi * _r) / MIN_CIRC_DISTANCE) + 1) for _r in radii]

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

comm_actions = radii
comm_states = np.unique(comm_states_arr, axis=0)
'''

"""
Utilities
"""


class RandomTrajectoriesGeneration(object):
    """
    Random trajectories generation
    """

    def __init__(self, source, destination,
                 radial_bounds, angular_bounds, swarm_size, segment_size, interpolation_num):
        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num

    def __enter__(self):
        return self

    def __generate(self, traj):
        m = self.m
        (r_min, r_max), (th_min, th_max) = self.r_bounds, self.theta_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)

        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)),
                                             tf.multiply(r, tf.math.sin(theta))],
                                            axis=1), validate_shape=True, use_locking=True)

    def generate(self):
        n, m = self.n, self.m
        trajs = tf.Variable(tf.zeros(shape=(int(n / 2), m, 2), dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
            [executor.submit(self.__generate, trajs[i, :]) for i in range(int(n / 2))]
        return trajs

    def __optimize(self, traj, opt_traj):
        m_ip = self.m_ip
        a = self.r_bounds[1]
        x_0, x_m = self.x_0, self.x_m
        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), (m_ip * len(i_s)), dtype=np.float64)

        tf.compat.v1.assign(opt_traj, tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
                            validate_shape=True, use_locking=True)

    def optimize(self, trajs):
        n, m_post = self.n, (self.m_ip * (self.m + 2))
        opt_trajs = tf.Variable(tf.zeros(shape=[int(n / 2), m_post, 2], dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
            [executor.submit(self.__optimize, trajs[i, :], opt_trajs[i, :]) for i in range(int(n / 2))]
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print('[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


class DeterministicTrajectoriesGeneration(object):
    """
    Deterministic trajectories generation
    """

    def __init__(self, source, destination,
                 radial_bounds, angular_bounds, swarm_size, segment_size, interpolation_num):
        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num

    def __enter__(self):
        return self

    def generate_optimize(self):
        a = self.r_bounds[1]
        x_0, x_m = self.x_0, self.x_m
        n, m_post = self.n, (self.m_ip * (self.m + 2))

        x_mid = tf.divide(tf.add(x_0, x_m), 2)
        x_mid_0 = tf.divide(tf.add(x_0, x_mid), 2)
        x_mid_m = tf.divide(tf.add(x_mid, x_m), 2)
        traj = tf.concat([x_mid_0, x_mid, x_mid_m], axis=0)

        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), m_post, dtype=np.float64)

        return tf.tile(tf.expand_dims(tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
            axis=0), multiples=[int(n / 2), 1, 1])

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print('[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


def evaluate_power_consumption(uav_flying_velocity):
    """
    UAV mobility power consumption
    """
    v, u_tip, v_0 = uav_flying_velocity, ROTOR_BLADE_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY
    p_1, p_2, p_3 = POWER_PROFILE_CONSTANT_1, POWER_PROFILE_CONSTANT_2, POWER_PROFILE_CONSTANT_3

    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_3 * (v ** 3)) + \
        (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)


def fz(z_):
    """
    f(Z)
    """
    b = CHANNEL_BANDWIDTH
    return b * np.log2(1 + (0.5 * (z_ ** 2)))


def marcum_q(df, nc, x):
    """
    Marcum-Q
    """
    return 1 - ncx2.cdf(x, df, nc)


def f_obj(z_, *args):
    """
    Bisection objective
    """
    df, nc, y = args

    f_z = fz(z_)
    q_m = marcum_q(df, nc, (y * (z_ ** 2)))

    ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
    ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf

    return -ln_f_z - ln_q_m


def bisect(f_, df, nc, y, low, high, tolerance):
    """
    Bisection method
    """
    args = (df, nc, y)
    assert tolerance is not None
    mid, converged, conf, conf_th = 0.0, False, 0, BISECTION_CONVERGENCE_CONFIDENCE

    while (not converged) or (conf < conf_th):
        mid = (high + low) / 2
        if (f_(low, *args) * f_(high, *args)) > 0.0:
            low = mid
        else:
            high = mid
        converged = abs(high - low) < tolerance
        conf += 1 if converged else -conf

    return mid


def z_var(gamma):
    """
    Variable for re-formulation in Z
    """
    b = CHANNEL_BANDWIDTH
    return np.sqrt(2 * ((2 ** (gamma / b)) - 1))


def u_var(gamma, d, los):
    """
    Variable for re-formulation in u
    """
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    alpha, alpha_, kappa = LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT
    return ((2 ** (gamma / b)) - 1) / (gamma_ * 1 if los else kappa * (d ** -alpha if los else -alpha_))


def evaluate_los_throughput(d, phi, r_star_los):
    """
    LoS throughput
    """
    k_1, k_2 = LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    k, alpha = k_1 * np.exp(k_2 * phi), LoS_PATH_LOSS_EXPONENT
    df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), BISECTION_METHOD_TOLERANCE

    z_star = bisect(f_obj, df, nc, y, 0,
                    z_var(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ * (d ** -alpha))), t)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)


def evaluate_nlos_throughput(d, r_star_nlos):
    """
    NLoS throughput
    """
    alpha_, t = NLoS_PATH_LOSS_EXPONENT, BISECTION_METHOD_TOLERANCE
    b, gamma_, kappa = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, NLoS_ATTENUATION_CONSTANT

    df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))

    z_star = bisect(f_obj, df, nc, y, 0,
                    z_var(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa * (d ** -alpha_))), t)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)


def calculate_adapted_throughput(d, phi, r_star_los, r_star_nlos):
    """
    Rate-adapted throughput
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.submit(evaluate_nlos_throughput, d, r_star_nlos)
        executor.submit(evaluate_los_throughput, d, phi, r_star_los)


max_power = evaluate_power_consumption(MAX_UAV_VELOCITY)
PENALTIES_CAPSULE = namedtuple('penalties_capsule', ['t_p_1', 't_p_2', 'e_p_1', 'e_p_2'])

"""
Core operations
"""


def interpolate_waypoints(p_indices, p, res_multiplier):
    m = len(p_indices)
    spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)

    return tf.constant(list(zip(spl_x(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)),
                                spl_y(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)))))


def interpolate_velocities(v_indices, v, res_multiplier):
    m, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
    return spl_v(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64))


def penalties(p__, v__, x_g, res_multiplier):
    min_power = evaluate_power_consumption(22.0)
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2

    p = interpolate_waypoints([_ for _ in range(p__.shape[0])], p__, res_multiplier)

    midpoint, h_uav, h_bs = int(p.shape[0] / 2), UAV_HEIGHT, BASE_STATION_HEIGHT

    v = interpolate_velocities([_ for _ in range(v__.shape[0])], v__, res_multiplier)

    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] -
                          p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    # Decode (GN --> UAV)

    r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
    h_u = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)

    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
    phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()

    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
            executor.submit(calculate_adapted_throughput, d_gu__, phi_gu__, r_los_gu[i__], r_nlos_gu[i__])

    phi_degrees_gu = (180.0 / np.pi) * phi_gu
    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

    h_1 = data_payload_size - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))

    t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
    e_p_1 = (lambda: 0.0, lambda: (min_power * t_p_1))[h_1.numpy() > 0.0]()

    # Forward (UAV --> BS)

    r_ub = tf.norm(p[midpoint:], axis=1)
    h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)

    d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
    phi_ub = tf.asin(tf.divide(h_ub, d_ub))

    r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=NUMBER_OF_WORKERS) as executor:
        for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
            executor.submit(calculate_adapted_throughput, d_ub__, phi_ub__, r_los_ub[i__], r_nlos_ub[i__])

    phi_degrees_ub = (180.0 / np.pi) * phi_ub
    p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
    p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
    r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))

    h_2 = data_payload_size - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))

    t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
    e_p_2 = (lambda: 0.0, lambda: (min_power * t_p_2))[h_2.numpy() > 0.0]()

    return PENALTIES_CAPSULE(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)


@tf.function
def power_cost(v):
    return tf.map_fn(evaluate_power_consumption, v, parallel_iterations=num_workers)


def calculate_comm_cost(p, v, x_g, f_hat, e_hat=None, t_hat=None):
    interp = INTERPOLATION_FACTOR
    t_p_1, t_p_2, e_p_1, e_p_2 = penalties(p, v, x_g, interp)
    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] -
                          p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    e__, t__ = tf.reduce_sum(tf.multiply(t, power_cost(v[:-1]))), tf.reduce_sum(t)

    tf.compat.v1.assign(f_hat, ((1.0 - (2.0 * h_alpha)) * (t_p_1 + t_p_2 + t__)) +
                        ((h_alpha / max_power) * (e_p_1 + e_p_2 + e__)), validate_shape=True, use_locking=True)

    if e_hat is not None:
        tf.compat.v1.assign(e_hat, e__, validate_shape=True, use_locking=True)

    if t_hat is not None:
        tf.compat.v1.assign(t_hat, t__, validate_shape=True, use_locking=True)


def update_winners_and_losers(p, v, u, w, t_j, t_j_1, p_bar, v_bar, f_hats, x_g):
    v_min, v_max, omega = CSO_MINIMUM_VELOCITY_VALUE, MAX_UAV_VELOCITY, CSO_PARTICLE_VELOCITY_SCALING_FACTOR

    f_hat_t_j, f_hat_t_j_1 = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
    p_t_j, p_t_j_1, v_t_j, v_t_j_1 = p[t_j], p[t_j_1], v[t_j], v[t_j_1]
    u_t_j, u_t_j_1, w_t_j, w_t_j_1 = u[t_j], u[t_j_1], w[t_j], w[t_j_1]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.submit(calculate_comm_cost, p_t_j, v_t_j, x_g, f_hat_t_j)
        executor.submit(calculate_comm_cost, p_t_j_1, v_t_j_1, x_g, f_hat_t_j_1)

    argmin__ = np.argmin([f_hat_t_j, f_hat_t_j_1])
    j_win, p_w, v_w, j_los, p_l, v_l, u_l, w_l = (t_j, p_t_j, v_t_j, t_j_1, p_t_j_1, v_t_j_1, u_t_j_1, w_t_j_1) \
        if (argmin__ == 0) else (t_j_1, p_t_j_1, v_t_j_1, t_j, p_t_j, v_t_j, u_t_j, w_t_j)

    r_j = tf.random.uniform(shape=[3, ], dtype=tf.float64)

    # Trajectory particles & associated particle velocities updates

    tf.compat.v1.assign(f_hats[t_j], f_hat_t_j, validate_shape=True, use_locking=True)

    u_j_los = tf.add_n([tf.multiply(r_j[0], u_l),
                        tf.multiply(r_j[1], tf.subtract(p_w, p_l)),
                        omega * tf.multiply(r_j[2], tf.subtract(p_bar, p_l))])

    p_j_los = tf.add(p_l, u_j_los)
    tf.compat.v1.assign(p[j_los], p_j_los, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(u[j_los], u_j_los, validate_shape=True, use_locking=True)

    # UAV velocity particles & associated particle velocities updates

    tf.compat.v1.assign(f_hats[t_j_1], f_hat_t_j_1, validate_shape=True, use_locking=True)

    w_j_los = tf.add_n([tf.multiply(r_j[0], w_l),
                        tf.multiply(r_j[1], tf.subtract(v_w, v_l)),
                        omega * tf.multiply(r_j[2], tf.subtract(v_bar, v_l))])

    v_j_los = tf.clip_by_value(tf.add(v_l, w_j_los), v_min, v_max)
    tf.compat.v1.assign(v[j_los], v_j_los, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(w[j_los], w_j_los, validate_shape=True, use_locking=True)


def competitive_swarm_optimization(initial_uav_pos, terminal_uav_pos,
                                   gn_pos, trajectory_particles, uav_velocity_particles,
                                   traj_particle_velocities, uav_vel_particle_velocities, swarm_size, segment_size):
    x_0, x_m, x_g = initial_uav_pos, terminal_uav_pos, gn_pos

    p, v = trajectory_particles, uav_velocity_particles
    p_bar, v_bar = tf.reduce_mean(p, axis=0), tf.reduce_mean(v, axis=0)

    u, w = traj_particle_velocities, uav_vel_particle_velocities
    k, n, m, k_max = 0, swarm_size, segment_size, MAXIMUM_COST_EVALUATIONS

    f_hats = tf.Variable(tf.zeros(shape=[n, ], dtype=tf.float64), dtype=tf.float64)
    indices = [_ for _ in range(n)]

    while k <= k_max:
        print(k)
        t = tf.random.shuffle(indices)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for j in range(0, n, 2):
                executor.submit(update_winners_and_losers, p, v, u, w, t[j], t[j + 1], p_bar, v_bar, f_hats, x_g)

        k += 1

    i = min(indices, key=lambda k__: f_hats[k__].numpy())
    return p[i], u[i], v[i], w[i], f_hats[i]


# noinspection PyUnusedLocal
def hierarchical_competitive_swarm_optimization(initial_uav_position, terminal_uav_position,
                                                ground_node_position, optimal_trajectories, optimal_velocities):
    i, f_hat = 0, tf.Variable(0.0, dtype=tf.float64)
    p_star, u_star, v_star, w_star = None, None, None, None

    a, v_levels = CELL_RADIUS, CSO_VELOCITY_DISCRETIZATION_LEVELS
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2
    v_min, v_max, eps = CSO_MINIMUM_VELOCITY_VALUE, MAX_UAV_VELOCITY, HCSO_VELOCITY_SCALING_FACTOR

    n, m_old, m_ip = INITIAL_NUMBER_OF_PARTICLES, INITIAL_TRAJECTORY_SEGMENTS, INTERPOLATION_FACTOR
    x_0, x_m, x_g, h_b = initial_uav_position, terminal_uav_position, ground_node_position, BASE_STATION_HEIGHT

    zeta, m, m_max = HCSO_TRAJECTORY_SCALING_FACTOR, m_old, MAXIMUM_TRAJECTORY_SEGMENTS
    assert m_max > m and m_max >= m * HCSO_VALIDATION_MULTIPLIER

    r_gen = RandomTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
    d_gen = DeterministicTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
    p, m = tf.concat([d_gen.generate_optimize(), r_gen.optimize(r_gen.generate())], axis=0), (m + 2) * m_ip

    velocity_vals = np.linspace(v_min, v_max, v_levels)

    u = tf.Variable(np.random.choice(velocity_vals, size=[n, m, 2]))
    v = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))
    w = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))

    # HCSO while loop
    while m <= m_max * m_ip:
        print(m)
        p_star, u_star, v_star, w_star, f_star = competitive_swarm_optimization(x_0, x_m, x_g, p, v, u, w, n, m)

        n -= 2 * (i + 1)

        i += 1
        indices = [_ for _ in range(m)]

        # Trajectory particles

        p_tilde = tf.tile(tf.expand_dims(interpolate_waypoints(indices, p_star, m_ip), axis=0), multiples=[n, 1, 1])
        p, p_ = tf.Variable(tf.zeros(shape=p_tilde.shape, dtype=tf.float64), dtype=tf.float64), p_tilde[:, :-1, :]

        u = tf.Variable(tf.zeros(shape=p_tilde.shape, dtype=tf.float64), dtype=tf.float64)

        p_1, p_2 = tf.roll(p_tilde, shift=-1, axis=1)[:, :-1, :], tf.roll(p_tilde, shift=1, axis=1)[:, 1:, :]

        scale = tf.expand_dims(tf.add(tf.square(tf.norm(tf.subtract(p_1, p_), axis=2)),
                                      tf.multiply(zeta, tf.square(tf.norm(tf.subtract(p_2, p_), axis=2)))), axis=2)

        scales, shape_p = tf.tile(scale, multiples=[1, 1, 2]), [n, (m_ip * m) - 1, 2]
        scale_last = tf.expand_dims(tf.add(tf.square(tf.norm(p_tilde[:, -1, :], axis=1)),
                                           tf.multiply(zeta, tf.square(tf.norm(tf.subtract(p_tilde[:, -2, :],
                                                                                           p_tilde[:, -1, :]),
                                                                               axis=1)))), axis=1)

        p_tilde_r = tf.random.normal(shape=shape_p, mean=tf.zeros(shape=shape_p, dtype=tf.float64),
                                     stddev=tf.sqrt(scales), dtype=tf.float64)

        tf.compat.v1.assign(p[:, :-1, :], tf.add(p_tilde[:, :-1, :], p_tilde_r), validate_shape=True, use_locking=True)

        tf.compat.v1.assign(p[:, -1, :], tf.add(p_tilde[:, -1, :],
                                                tf.random.normal(shape=[scale_last.shape[0], 2],
                                                                 mean=tf.zeros(shape=[scale_last.shape[0], 2],
                                                                               dtype=tf.float64),
                                                                 stddev=tf.tile(tf.sqrt(scale_last),
                                                                                multiples=[1, 2]), dtype=tf.float64)),
                            validate_shape=True, use_locking=True)

        tf.compat.v1.assign(u, tf.tile(tf.expand_dims(interpolate_waypoints(indices, u_star, m_ip), axis=0),
                                       multiples=[n, 1, 1]), validate_shape=True, use_locking=True)

        # UAV velocity particles

        shape_v = [n, m_ip * m]

        v_tilde = tf.tile(tf.expand_dims(interpolate_velocities(indices, v_star, m_ip), axis=0), multiples=[n, 1])

        v = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
        w = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)

        v_tilde_r = tf.random.normal(shape=shape_v, mean=tf.zeros(shape=shape_v, dtype=tf.float64),
                                     stddev=tf.sqrt(tf.multiply(eps * ((v_max - v_min) ** 2),
                                                                tf.ones(shape=shape_v, dtype=tf.float64))),
                                     dtype=tf.float64)

        tf.compat.v1.assign(v, tf.clip_by_value(tf.add(v_tilde, v_tilde_r), v_min, v_max),
                            validate_shape=True, use_locking=True)

        tf.compat.v1.assign(w, tf.tile(tf.expand_dims(interpolate_velocities(indices, w_star, m_ip), axis=0),
                                       multiples=[n, 1]), validate_shape=False, use_locking=True)

        m *= m_ip

    tf.compat.v1.assign(optimal_velocities, v_star, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(optimal_trajectories, p_star, validate_shape=True, use_locking=True)


# Run Trigger
if __name__ == '__main__':
    output_dir, data_payload_size = OUTPUT_DIR, DATA_PAYLOAD_SIZE
    h_alpha, seq_num, num_workers = HCSO_METRIC_ALPHA, 0, NUMBER_OF_WORKERS

    print('[INFO] MAESTROTrajectoryDesign main: Starting MAESTRO Trajectory Design - '
          f'Data Payload Size [L] = {data_payload_size / 1e6} Mb | HCSO Metric [alpha] = {h_alpha}.')

    comm_action = 250.0
    comm_state = [500.0, 625.0, 2.0478210642251344]

    r_u_ = comm_action
    r_u, r_gn, psi = comm_state
    x_init = tf.constant([[500.0, 0.0]], dtype=tf.float64)
    x_final = tf.constant([[177.0, 177.0]], dtype=tf.float64)
    x_gn = tf.constant([[193.0, 594.0]], dtype=tf.float64)

    o_trajs = tf.Variable(tf.zeros(shape=[INTERPOLATION_FACTOR * MAXIMUM_TRAJECTORY_SEGMENTS, 2],
                                   dtype=tf.float64), dtype=tf.float64)

    o_velocities = tf.Variable(tf.zeros(shape=[INTERPOLATION_FACTOR * MAXIMUM_TRAJECTORY_SEGMENTS, ],
                                        dtype=tf.float64), dtype=tf.float64)

    hierarchical_competitive_swarm_optimization(x_init, x_final, x_gn, o_trajs, o_velocities)

    tf.io.write_file(f'{output_dir}{seq_num}.log',
                     tf.strings.format('{}\n{}\n{}\n{}',
                                       (tf.constant(str(comm_state), dtype=tf.string),
                                        tf.constant(str(comm_action), dtype=tf.string),
                                        tf.constant(str(o_velocities.numpy()), dtype=tf.string),
                                        tf.constant(str(o_trajs.numpy()), dtype=tf.string))), name=f'{h_alpha}')

    print('[INFO] MAESTROTrajectoryDesign main: MAESTRO Trajectory Design has been completed - '
          f'Data Payload Size [L] = {data_payload_size / 1e6} Mb | HCSO Metric [alpha] = {h_alpha}.')
