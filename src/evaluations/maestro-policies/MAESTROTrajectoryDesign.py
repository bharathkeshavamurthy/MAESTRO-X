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
Configurations-I: Logging setup for TensorFlow | XLA-JIT enhancement for the ASU-EXXACT GPU cluster 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
                             '/home/bkeshav1/workspace/repos/MAESTRO-X/src/maestro-policies/MAESTROTrajectoryDesign.py'

import time
import plotly
import warnings
import traceback
import numpy as np
import tensorflow as tf
from collections import namedtuple
import plotly.graph_objs as graph_objs
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# NumPy seed
np.random.seed(6)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Plotly API settings
"""
plotly.tools.set_credentials_file(username='bkeshav1', api_key='PUYaTVhV1Ok04I07S4lU')

"""
Configurations-III: Simulation parameters
"""

'''Deployment Model'''

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1e3

''' Motion and Power Consumption Model '''

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

''' Channel Model '''

# The additional NLoS attenuation factor ($\kappa$)
NLoS_ATTENUATION_CONSTANT = 0.2

# The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
LoS_PATH_LOSS_EXPONENT = 2.0

# The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
NLoS_PATH_LOSS_EXPONENT = 2.8

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

# The propagation environment specific parameter ($z_{1}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61

# The propagation environment specific parameter ($z_{2}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16

# The propagation environment dependent coefficient ($k_{1}$) for the LoS Rician link model's $K$-factor
LoS_RICIAN_FACTOR_1 = 1.0

# The propagation environment dependent coefficient ($k_{2}$) for the LoS Rician link model's $K$-factor
LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0

# The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$) [40 dB]
REFERENCE_SNR_AT_1_METER = linear((5e6 * 40) / CHANNEL_BANDWIDTH)

''' Algorithmic Model '''

# The convergence confidence level for optimization algorithms in this framework
CONVERGENCE_CONFIDENCE = 10

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

# The initial number of trajectory segments in the HCSO algorithm ($M$)
INITIAL_TRAJECTORY_SEGMENTS = 6

# The UAV velocity and CSO/HCSO particle velocity discretization levels
CSO_VELOCITY_DISCRETIZATION_LEVELS = 25

# The smallest UAV velocity and CSO/HCSO particle velocity value ($V_{\text{low}}$)
CSO_MINIMUM_VELOCITY_VALUE = 0.0

# The maximum number of trajectory segments allowed in the HCSO solution ($M_{\text{max}}$)
MAXIMUM_TRAJECTORY_SEGMENTS = 128

# The number of initial trajectory and UAV velocity particles in the HCSO algorithm ($N$) [Swarm Size]
INITIAL_NUMBER_OF_PARTICLES = 400

# The maximum number of cost function evaluations recommended in the CSO algorithm ($\math{N}_{\text{max}}$)
MAXIMUM_COST_EVALUATIONS = 100

# A validation multiplier to make sure that the HCSO trajectory parameters are valid vis-à-vis the algorithm
HCSO_VALIDATION_MULTIPLIER = 10

# The number of waypoints to be interpolated between any two given points in the generated $M$-segment trajectories
INTERPOLATION_FACTOR = 2

# The scaling factor employed in the "disturbance around a trajectory reference solution" aspect of HCSO ($\zeta$)
HCSO_TRAJECTORY_SCALING_FACTOR = 1.0

# The scaling factor employed in the "disturbance around a velocity reference solution" aspect of HCSO ($\epsilon$)
HCSO_VELOCITY_SCALING_FACTOR = 1.0

# The scaling factor that determines the degree of influence of the global means in the CSO algorithm ($\omega$)
CSO_PARTICLE_VELOCITY_SCALING_FACTOR = 1.0

''' Visualization settings '''

# The Plotly API "markers-only" scatter plot mode
PLOTLY_MARKERS_MODE = 'markers'

# The Plotly API "lines and markers" scatter plot mode
PLOTLY_LINES_MARKERS_MODE = 'lines+markers'

"""
Configurations-IV: Deployments (BS, GNs, and UAVs)
"""
average_power_constraint = 1.2e3
gn_pos = tf.constant([[-570.0, 601.0]], dtype=tf.float64)
dual_var, packet_length = 0.99 / average_power_constraint, 10e6
uav_start_pos = tf.constant([[400.0, -300.0]], dtype=tf.float64)
uav_end_pos = tf.constant([[-387.50, 391.50]], dtype=tf.float64)

"""
Utilities
"""

# A namedtuple constituting all the relevant penalty metrics involved in the CSO/HCSO cost function evaluation
PENALTIES_CAPSULE = namedtuple('penalties_capsule', ['t_p_1', 't_p_2', 'e_p_1', 'e_p_2'])


# Random Trajectories Generation for HCSO Initialization
class RandomTrajectoriesGeneration(object):
    """
    Random Trajectories Generation
    """

    def __init__(self, source, destination, radial_bounds, angular_bounds, swarm_size,
                 segment_size, interpolation_num):
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

        with ThreadPoolExecutor(max_workers=1024) as executor:
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

        with ThreadPoolExecutor(max_workers=1024) as executor:
            [executor.submit(self.__optimize, trajs[i, :], opt_trajs[i, :]) for i in range(int(n / 2))]
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


# Deterministic Trajectories Generation for HCSO Initialization
class DeterministicTrajectoriesGeneration(object):
    """
    Deterministic Trajectories Generation
    """

    def __init__(self, source, destination, radial_bounds,
                 angular_bounds, swarm_size, segment_size, interpolation_num):
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
            print(f'[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


def __evaluate_power_consumption(uav_flying_velocity):
    """
    UAV mobility power consumption
    """
    v, u_tip, v_0 = uav_flying_velocity, ROTOR_BLADE_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY
    p_1, p_2, p_3 = POWER_PROFILE_CONSTANT_1, POWER_PROFILE_CONSTANT_2, POWER_PROFILE_CONSTANT_3

    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_3 * (v ** 3)) + \
        (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)


def __f_z(z):
    """
    f(z)
    """
    b = CHANNEL_BANDWIDTH
    return b * np.log2(1 + (0.5 * (z ** 2)))


def __marcum_q(df, nc, x):
    """
    Marcum-Q
    """
    return 1 - ncx2.cdf(x, df, nc)


def __f(z, *args):
    """
    Bisection objective
    """
    df, nc, y = args
    f_z, q_m = __f_z(z), __marcum_q(df, nc, (y * (z ** 2)))

    ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
    ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf

    return -ln_f_z - ln_q_m


def __bisect(f, df, nc, y, low, high, tolerance):
    """
    Bisection
    """
    args = (df, nc, y)
    assert tolerance is not None
    mid, converged, conf, conf_th = 0.0, False, 0, CONVERGENCE_CONFIDENCE

    while (not converged) or (conf < conf_th):
        mid = (high + low) / 2
        if (f(low, *args) * f(high, *args)) > 0.0:
            low = mid
        else:
            high = mid
        converged = (abs(high - low) < tolerance)
        conf += 1 if converged else -conf

    return mid


def __z(gamma):
    """
    Objective variable (z)
    """
    b = CHANNEL_BANDWIDTH
    return np.sqrt(2 * ((2 ** (gamma / b)) - 1))


def __u(gamma, d, los):
    """
    Transient change-of-variables (u)
    """
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    alpha, alpha_, kappa = LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT
    return ((2 ** (gamma / b)) - 1) / (gamma_ * 1 if los else kappa * (d ** -alpha if los else -alpha_))


def __evaluate_los_throughput(d, phi, r_star_los):
    """
    Adapted LoS throughput
    """
    k_1, k_2 = LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    k, alpha = k_1 * np.exp(k_2 * phi), LoS_PATH_LOSS_EXPONENT
    df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), BISECTION_METHOD_TOLERANCE

    z_star = __bisect(__f, df, nc, y, 0,
                      __z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ * (d ** -alpha))), t)

    gamma_star = __f_z(z_star)
    tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)


def __evaluate_nlos_throughput(d, r_star_nlos):
    """
    Adapted NLoS throughput
    """
    alpha_, t = NLoS_PATH_LOSS_EXPONENT, BISECTION_METHOD_TOLERANCE
    b, gamma_, kappa = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, NLoS_ATTENUATION_CONSTANT

    df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))

    z_star = __bisect(__f, df, nc, y, 0,
                      __z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa * (d ** -alpha_))), t)

    gamma_star = __f_z(z_star)
    tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)


def __calculate_adapted_throughput(d, phi, r_star_los, r_star_nlos):
    """
    Adapted throughput
    """
    with ThreadPoolExecutor(max_workers=1024) as executor:
        executor.submit(__evaluate_nlos_throughput, d, r_star_nlos)
        executor.submit(__evaluate_los_throughput, d, phi, r_star_los)


"""
Core operations
"""


def __interpolate_waypoints(p_indices, p, res_multiplier):
    m = len(p_indices)
    spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)

    return tf.constant(list(zip(spl_x(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)),
                                spl_y(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)))))


def __interpolate_velocities(v_indices, v, res_multiplier):
    m, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
    return spl_v(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64))


def __penalties(p__, v__, x_g, res_multiplier):
    min_power = __evaluate_power_consumption(22.0)
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2

    p = __interpolate_waypoints([_ for _ in range(p__.shape[0])], p__, res_multiplier)

    midpoint, h_uav, h_bs = int(p.shape[0] / 2), UAV_HEIGHT, BASE_STATION_HEIGHT

    v = __interpolate_velocities([_ for _ in range(v__.shape[0])], v__, res_multiplier)

    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] -
                          p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    # Decode (GN --> UAV)

    r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
    h_u = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)

    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
    phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()

    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=1024) as executor:
        for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
            executor.submit(__calculate_adapted_throughput, d_gu__, phi_gu__, r_los_gu[i__], r_nlos_gu[i__])

    phi_degrees_gu = (180.0 / np.pi) * phi_gu
    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

    h_1 = packet_length - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))

    t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
    e_p_1 = (lambda: 0.0, lambda: (min_power * t_p_1))[h_1.numpy() > 0.0]()

    # Forward (UAV --> BS)

    r_ub = tf.norm(p[midpoint:], axis=1)
    h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)

    d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
    phi_ub = tf.asin(tf.divide(h_ub, d_ub))

    r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=1024) as executor:
        for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
            executor.submit(__calculate_adapted_throughput, d_ub__, phi_ub__, r_los_ub[i__], r_nlos_ub[i__])

    phi_degrees_ub = (180.0 / np.pi) * phi_ub
    p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
    p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
    r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))

    h_2 = packet_length - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))

    t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
    e_p_2 = (lambda: 0.0, lambda: (min_power * t_p_2))[h_2.numpy() > 0.0]()

    return PENALTIES_CAPSULE(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)


@tf.function
def __power_cost(v, num_workers=1024):
    return tf.map_fn(__evaluate_power_consumption, v, parallel_iterations=num_workers)


def __calculate_comm_cost(p, v, nu, x_g, f_hat, e_hat=None, t_hat=None):
    p_average, interp = average_power_constraint, INTERPOLATION_FACTOR

    t_p_1, t_p_2, e_p_1, e_p_2 = __penalties(p, v, x_g, interp)
    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] -
                          p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    e__, t__ = tf.reduce_sum(tf.multiply(t, __power_cost(v[:-1]))), tf.reduce_sum(t)

    tf.compat.v1.assign(f_hat, ((1.0 - (nu * p_average)) * (t_p_1 + t_p_2 + t__)) +
                        (nu * (e_p_1 + e_p_2 + e__)), validate_shape=True, use_locking=True)

    if e_hat is not None:
        tf.compat.v1.assign(e_hat, e__, validate_shape=True, use_locking=True)

    if t_hat is not None:
        tf.compat.v1.assign(t_hat, t__, validate_shape=True, use_locking=True)


def __update_winners_and_losers(p, v, u, w, t_j, t_j_1, p_bar, v_bar, f_hats, nu, x_g):
    v_min, v_max, omega = CSO_MINIMUM_VELOCITY_VALUE, MAX_UAV_VELOCITY, CSO_PARTICLE_VELOCITY_SCALING_FACTOR

    f_hat_t_j, f_hat_t_j_1 = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
    p_t_j, p_t_j_1, v_t_j, v_t_j_1 = p[t_j], p[t_j_1], v[t_j], v[t_j_1]
    u_t_j, u_t_j_1, w_t_j, w_t_j_1 = u[t_j], u[t_j_1], w[t_j], w[t_j_1]

    with ThreadPoolExecutor(max_workers=1024) as executor:
        executor.submit(__calculate_comm_cost, p_t_j, v_t_j, nu, x_g, f_hat_t_j)
        executor.submit(__calculate_comm_cost, p_t_j_1, v_t_j_1, nu, x_g, f_hat_t_j_1)

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


def __competitive_swarm_optimization(initial_uav_position, terminal_uav_position, ground_node_position,
                                     trajectory_particles, uav_velocity_particles, traj_particle_velocities,
                                     uav_velocity_particle_velocities, swarm_size, segment_size, dual_variable):
    x_0, x_m, x_g = initial_uav_position, terminal_uav_position, ground_node_position

    p, v = trajectory_particles, uav_velocity_particles
    p_bar, v_bar = tf.reduce_mean(p, axis=0), tf.reduce_mean(v, axis=0)

    u, w = traj_particle_velocities, uav_velocity_particle_velocities
    k, n, m, nu, k_max = 0, swarm_size, segment_size, dual_variable, MAXIMUM_COST_EVALUATIONS

    f_hats = tf.Variable(tf.zeros(shape=[n, ], dtype=tf.float64), dtype=tf.float64)
    indices = [_ for _ in range(n)]

    while k <= k_max:
        t = tf.random.shuffle(indices)

        with ThreadPoolExecutor(max_workers=1024) as executor:
            for j in range(0, n, 2):
                executor.submit(__update_winners_and_losers, p, v, u, w, t[j], t[j + 1], p_bar, v_bar, f_hats, nu, x_g)

        k += 1

    i = min(indices, key=lambda k__: f_hats[k__].numpy())
    return p[i], u[i], v[i], w[i], f_hats[i]


# noinspection PyUnusedLocal
def hierarchical_competitive_swarm_optimization(initial_uav_position, terminal_uav_position,
                                                dual_variable, ground_node_position, lagrangian, energy, time_duration):
    xi_s, opt_traj = relay_status, optimal_trajectories
    p_star, u_star, v_star, w_star = None, None, None, None
    i, nu, f_hat = 0, dual_variable, tf.Variable(0.0, dtype=tf.float64)

    _comm_star, e_comm_star, t_comm_star = lagrangian, energy, time_duration
    l_comm_star, e_comm_star, t_comm_star = lagrangian, energy, time_duration
    e_usage, delta = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2
    a, v_levels, p_len = CELL_RADIUS, CSO_VELOCITY_DISCRETIZATION_LEVELS, packet_length
    v_min, v_max, eps = CSO_MINIMUM_VELOCITY_VALUE, MAX_UAV_VELOCITY, HCSO_VELOCITY_SCALING_FACTOR

    n, m_old, m_ip = INITIAL_NUMBER_OF_PARTICLES, INITIAL_TRAJECTORY_SEGMENTS, INTERPOLATION_FACTOR
    x_0, x_m, x_g, h_b = initial_uav_position, terminal_uav_position, ground_node_position, BASE_STATION_HEIGHT

    zeta, m, m_max = HCSO_TRAJECTORY_SCALING_FACTOR, m_old, MAXIMUM_TRAJECTORY_SEGMENTS
    assert m_max > m and m_max >= m * HCSO_VALIDATION_MULTIPLIER

    r_gen = RandomTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
    d_gen = DeterministicTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
    p, m = tf.concat([d_gen.generate_optimize(), r_gen.optimize(r_gen.generate())], axis=0), ((m + 2) * m_ip)

    m = ((m + 2) * m_ip)
    velocity_vals = np.linspace(v_min, v_max, v_levels)

    u = tf.Variable(np.random.choice(velocity_vals, size=[n, m, 2]))
    v = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))
    w = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))

    # HCSO while loop
    while m <= m_max * m_ip:
        p_star, u_star, v_star, w_star, f_star = __competitive_swarm_optimization(x_0, x_m, x_g, p, v, u, w, n, m, nu)

        comm_costs[time.monotonic()] = f_star
        n -= 20 * (i + 1)

        i += 1
        indices = [_ for _ in range(m)]

        # Trajectory particles

        p_tilde = tf.tile(tf.expand_dims(__interpolate_waypoints(indices, p_star, m_ip), axis=0), multiples=[n, 1, 1])
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

        tf.compat.v1.assign(u, tf.tile(tf.expand_dims(__interpolate_waypoints(indices, u_star, m_ip), axis=0),
                                       multiples=[n, 1, 1]), validate_shape=True, use_locking=True)

        # UAV velocity particles

        shape_v = [n, m_ip * m]

        v_tilde = tf.tile(tf.expand_dims(__interpolate_velocities(indices, v_star, m_ip), axis=0), multiples=[n, 1])

        v = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
        w = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)

        v_tilde_r = tf.random.normal(shape=shape_v, mean=tf.zeros(shape=shape_v, dtype=tf.float64),
                                     stddev=tf.sqrt(tf.multiply(eps * ((v_max - v_min) ** 2),
                                                                tf.ones(shape=shape_v, dtype=tf.float64))),
                                     dtype=tf.float64)

        tf.compat.v1.assign(v, tf.clip_by_value(tf.add(v_tilde, v_tilde_r), v_min, v_max),
                            validate_shape=True, use_locking=True)

        tf.compat.v1.assign(w, tf.tile(tf.expand_dims(__interpolate_velocities(indices, w_star, m_ip), axis=0),
                                       multiples=[n, 1]), validate_shape=False, use_locking=True)

        m *= m_ip

    # Lagrangian cost determination for scheduling (Direct BS or UAV relay?)

    d_gb = np.sqrt(np.add(np.square(h_b), np.square(np.linalg.norm(x_g))))
    phi_gb = np.arcsin(tf.divide(h_b, d_gb))

    r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=1024) as executor:
        executor.submit(__calculate_adapted_throughput, d_gb, phi_gb, r_los_gb, r_nlos_gb)
        executor.submit(__calculate_comm_cost, p_star, v_star, nu, x_g, f_hat, e_usage, delta)

    phi_degrees_gb = (180.0 / np.pi) * phi_gb

    p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
    p_nlos_gb = 1 - p_los_gb

    r_bar_gb = tf.add(tf.multiply(p_los_gb, r_los_gb), tf.multiply(p_nlos_gb, r_nlos_gb))

    l_xi_0 = (p_len / r_bar_gb) if r_bar_gb != 0.0 else np.inf
    l_xi_1, e_xi_1, t_xi_1 = f_hat.numpy(), e_usage.numpy(), delta.numpy()
    l__, e__, t__, xi__ = (l_xi_1, e_xi_1, t_xi_1, 1) if (l_xi_1 < l_xi_0) else (l_xi_0, 0.0, l_xi_0, 0)

    # Core metrics for debugging and visualization
    xi_s = xi__
    tf.compat.v1.assign(l_comm_star, l__, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(e_comm_star, e__, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(t_comm_star, t__, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(opt_traj, p_star, validate_shape=True, use_locking=True)


# Run Trigger
if __name__ == '__main__':
    print('[INFO] MAESTROTrajectoryDesign main: Starting Evaluation - '
          f'Packet Length Constraint [L] = {packet_length / 1e6} Mbits | '
          f'Avg Power Constraint [P_avg] = {average_power_constraint / 1e3} kW.')

    relay_status, comm_costs, traj_plot_data, lagr = 0, dict(), list(), tf.Variable(0.0, dtype=tf.float64)
    nrg, delay = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

    optimal_trajectories = tf.Variable(tf.zeros(shape=[
        INTERPOLATION_FACTOR * MAXIMUM_TRAJECTORY_SEGMENTS, 2], dtype=tf.float64), dtype=tf.float64)

    hierarchical_competitive_swarm_optimization(uav_start_pos, uav_end_pos, dual_var, gn_pos, lagr, nrg, delay)

    print(f'[INFO] MAESTROTrajectoryDesign main: Relay Status = {relay_status} | '
          f'Lagrangian Cost = {lagr} | Energy Cost = {nrg} | Delay Cost = {delay}.')

    # Lagrangian cost convergence visualization

    t_vals = list(comm_costs.keys())
    lagr_vals = list(comm_costs.values())
    plot_layout = dict(title='Lagrangian Cost Convergence',
                       xaxis=dict(title='Time (in s)', type='log', autorange=True),
                       yaxis=dict(title='Lagrangian Cost', type='log', autorange=True))
    plot_data = graph_objs.Scatter(x=t_vals, y=lagr_vals, mode=PLOTLY_LINES_MARKERS_MODE)

    fig = dict(data=[plot_data], layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='Lagrangian_Cost_Convergence', auto_open=False)
    print('[INFO] MAESTROTrajectoryDesign main: The plot of the Lagrangian Cost versus Computation Time '
          f'for the UAV at {uav_start_pos[0].numpy()} and GN at {gn_pos[0].numpy()} is available at - {fig_url}.')

    # The optimal HCSO-determined UAV trajectory visualization

    traj_plot_data.append(graph_objs.Scatter(mode=PLOTLY_MARKERS_MODE,
                                             x=[uav_start_pos[0, 0].numpy()],
                                             y=[uav_start_pos[0, 1].numpy()],
                                             name='UAV Initial Position ' + str(uav_start_pos[0].numpy())))

    traj_plot_data.append(graph_objs.Scatter(x=[gn_pos[0, 0].numpy()], y=[gn_pos[0, 1].numpy()],
                                             mode=PLOTLY_MARKERS_MODE, name='GN Position ' + str(gn_pos[0].numpy())))

    traj_plot_data.append(graph_objs.Scatter(x=optimal_trajectories[:, 0].numpy(),
                                             y=optimal_trajectories[:, 1].numpy(),
                                             mode=PLOTLY_LINES_MARKERS_MODE, name='UAV Optimal Trajectory'))

    plot_layout = dict(title='Optimal HCSO-determined UAV Trajectory',
                       xaxis=dict(title='x (in m)'), yaxis=dict(title='y (in m)'))

    fig = dict(data=traj_plot_data, layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='HCSO_UAV_Trajectory', auto_open=False)
    print('[INFO] MAESTROTrajectoryDesign main: The plot of the optimal HCSO-determined UAV trajectory '
          f'for the GN at {gn_pos[0].numpy()} [m, m], with Initial UAV Position = {uav_start_pos[0].numpy()} '
          f'[m, m] and Terminal UAV Position = {uav_end_pos[0].numpy()} [m, m], is available at - {fig_url}.')
