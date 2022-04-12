"""
This script encapsulates the following algorithms employed in MAESTRO (our adaptive scheduling & trajectory design
scheme for multi-winged, power-constrained, rotary UAVs):
    1. The value iteration algorithm for the SMDP formulation,
    2. The projected sub-gradient ascent algorithm for dual variable maximization,
    3. The Hierarchical Competitive Swarm Optimization (HCSO) algorithm for optimal trajectory determination, and
    4. The Competitive Swarm Optimization (CSO) algorithm for lower-level trajectory optimization (driven by HCSO).

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Version v21.12: Mathematical Modeling by Matthew Bliss <mbliss@purdue.edu> and Nicolo Michelusi <michelus@purdue.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
Copyright (c) 2021. All Rights Reserved.
"""

"""
Release: Final CPU version, with Progress Bar: No device placement | No visualizations | Logging key metrics
"""

# The imports
import os

"""
Configuration-I: Logging setup for TensorFlow
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
                             '/home/bkeshav1/workspace/repos/Odin/src/uav-mobility/core/MAESTRO.py'

import sys
import math
import uuid
import warnings
import itertools
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
Global Settings
"""

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Numpy random seed | Numpy print options
np.random.seed(8)
np.set_printoptions(threshold=sys.maxsize)

# A global resource access semaphore for concurrent evaluations and analysis ops on shared visualization data objects
lock = Lock()

"""
Progress Print Routine
"""


def sys_progress(current_value, end_value, entity, routine, log_level, key, bar_length):
    """
    Visualize the progress using the sys.stdout.write() and sys.stdout.flush() routines

    :param current_value: The current status of the process being monitored
    :param end_value: The final status of the process being monitored [for normalization / completion-% evaluation]
    :param entity: The entity whose process is being monitored (class name)
    :param routine: The routine/process being monitored (method name)
    :param log_level: The log level of the process' progress monitoring
    :param key: An additional identifier key for better description of the logs
    :param bar_length: The length of the progress bar to be printed
    """
    arrow_position = '-' * int(((current_value / end_value) * bar_length) - 1) + '>'
    empty_spaces = ' ' * int((bar_length - len(arrow_position)))
    sys.stdout.write('\r[{}] {} {}: {} - [{}] {}%'.format(log_level, entity, routine, key,
                                                          arrow_position + empty_spaces,
                                                          np.round((current_value / end_value) * 100, 2)))
    sys.stdout.flush()


"""
Internal Random Optimized Trajectories Generation Class 
"""


class RandomTrajectoriesGeneration(object):
    """
    This class encapsulates the operations associated with generating a set of random trajectories (size=swarm_size $N$)
    between two given points (in rectangular coordinates: $\mathbf{x}_{0}$ and $\mathbf{x}_{M}$).
    """

    def __init__(self, source, destination, radial_bounds, angular_bounds, swarm_size,
                 segment_size, interpolation_num):
        """
        The initialization sequence

        :param source: The starting point for the random paths that are to be generated
                       $\mathbf{x}_{0}{=}(x_{0}, y_{0})$
        :param destination: The ending point for the random paths that are to be generated
                            $\mathbf{x}_{M}{=}(x_{M}, y_{M})$
        :param radial_bounds: The lower and upper bounds for the allowed radii values $(-a, a)$ [Polar Perspective]
        :param angular_bounds: The lower and upper bounds for the allowed angle values $(0, 2\pi)$ [Polar Perspective]
        :param swarm_size: The swarm size, i.e., the number of random trajectories to be generated [concurrently]
                           between x_0 and x_m ($N$)
        :param segment_size: The resolution of the line-segment (trajectory) generated between x_0 and x_m ($M$)
        :param interpolation_num: The number of points to be inserted/interpolated between any two given points in the
                                  generated random trajectory, during the "optimization via interpolation" phase
        """
        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    def __generate(self, traj):
        """
        Generate a random $M$-segment trajectory

        :param traj: Assign the generated random $M$-segment trajectory to this [indexed] collection [Mx2 tensor]
        """
        m = self.m
        (r_min, r_max), (th_min, th_max) = self.r_bounds, self.theta_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)
        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)), tf.multiply(r, tf.math.sin(theta))],
                                            axis=1), validate_shape=True, use_locking=True)

    def generate(self):
        """
        Generate $N$ (self.n) random $M$-segment (self.m) trajectories between $\mathbf{x}_{0}$ (self.x_0) and
        $\mathbf{x}_{M}$ (self.x_m) [Concurrent Ops | Pre-Optimization]

        :return: An (N, M, 2) tensor: $N$ $M$-trajectory segments with each point represented as a 2-tuple (x, y) in
                 rectangular coordinates
        """
        n, m = self.n, self.m
        trajs = tf.Variable(tf.zeros(shape=(int(n / 2), m, 2), dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=256) as executor:
            for i in range(int(n / 2)):
                executor.submit(self.__generate, trajs[i, :])
        return trajs

    def __optimize(self, traj, opt_traj):
        """
        Optimize the randomness for the given $M$-segment trajectory via SciPy UnivariateSpline Interpolation

        :param traj: The [indexed] random trajectory generated via uniform sampling in polar coordinates [Mx2 tensor]
        :param opt_traj: The [indexed] new trajectory collection to house the interpolated one [Concurrency Op]
                         [(m_ip * (M + 2))x2 tensor]
        """
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
        """
        Optimize the randomness via SciPy UnivariateSpline Interpolation

        :param trajs: The indexed random trajectories generated via uniform sampling in polar coordinates [NxMx2 tensor]

        :return The new trajectories' collection housing the optimized ones [Nx(m_ip * (M + 2))x2 tensor]
                [Concurrency Op]
        """
        n, m_post = self.n, (self.m_ip * (self.m + 2))
        opt_trajs = tf.Variable(tf.zeros(shape=[int(n / 2), m_post, 2], dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=256) as executor:
            for i in range(int(n / 2)):
                executor.submit(self.__optimize, trajs[i, :], opt_trajs[i, :])
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        if exc_tb is not None:
            print(f'[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Exception Type = {exc_type} | Exception Value = {exc_val} | '
                  f'Traceback = {traceback.print_tb(exc_tb)}')


"""
Deterministic Trajectories Generation
"""


class DeterministicTrajectoriesGeneration(object):
    """
    This class encapsulates the operations associated with generating a set of fixed trajectories (size=swarm_size $N$)
    between two given points (in rectangular coordinates: $\mathbf{x}_{0}$ and $\mathbf{x}_{M}$).
    """

    def __init__(self, source, destination, radial_bounds, angular_bounds, swarm_size,
                 segment_size, interpolation_num):
        """
        The initialization sequence

        :param source: The starting point for the deterministic paths that are to be generated
                       $\mathbf{x}_{0}{=}(x_{0}, y_{0})$
        :param destination: The ending point for the deterministic paths that are to be generated
                            $\mathbf{x}_{M}{=}(x_{M}, y_{M})$
        :param radial_bounds: The lower and upper bounds for the allowed radii values $(-a, a)$ [Polar Perspective]
        :param angular_bounds: The lower and upper bounds for the allowed angle values $(0, 2\pi)$ [Polar Perspective]
        :param swarm_size: The swarm size, i.e., the number of deterministic trajectories to be generated [concurrently]
                           between x_0 and x_m ($N$)
        :param segment_size: The resolution of the line-segment (trajectory) generated between x_0 and x_m ($M$)
        :param interpolation_num: The number of points to be inserted/interpolated between any two given points in the
                                  generated deterministic trajectory, during the "optimization via interpolation" phase
        """
        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    def generate_optimize(self):
        """
        Generate & Optimize $N$ (self.n) fixed $M$-segment (self.m) trajectories between $\mathbf{x}_{0}$ (self.x_0)
        and $\mathbf{x}_{M}$ (self.x_m) [Concurrent Ops]

        :return: An (N, M, 2) tensor: $N$ $M$-trajectory segments with each point represented as a 2-tuple (x, y) in
                 rectangular coordinates
        """
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
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        if exc_tb is not None:
            print(f'[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Exception Type = {exc_type} | Exception Value = {exc_val} | '
                  f'Traceback = {traceback.print_tb(exc_tb)}')


"""
The SMDP-HCSO Evaluation Class
"""


class MAESTRO(object):
    """
    This class constitutes the optimal scheduling & trajectory determination for a multi-winged, power-constrained,
    rotary Unmanned Aerial Vehicle (UAV), in settings with a Base Station (BS) at the center of a circular cell, dotted
    with a myriad of uniformly-distributed, heterogeneous, Ground Nodes (GNs) generating communication requests
    according to a Poisson process.
    """

    """
    Configurations-II: SYSTEM MODEL
    """

    """
    Channel Model Parameters
    """

    # The bandwidth of each orthogonal channel assigned to this application ($B$) in Hz
    CHANNEL_BANDWIDTH = 5e6

    # The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
    LoS_PATH_LOSS_EXPONENT = 2.0

    # The propagation environment dependent coefficient ($k_{1}$) for the LoS Rician link model's $K$-factor
    LoS_RICIAN_FACTOR_1 = 1.0

    # The propagation environment dependent coefficient ($k_{2}$) for the LoS Rician link model's $K$-factor
    LoS_RICIAN_FACTOR_2 = np.log(100) / 90.0

    # The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
    NLoS_PATH_LOSS_EXPONENT = 2.8

    # The additional NLoS attenuation factor ($\kappa$)
    NLoS_ATTENUATION_CONSTANT = 0.2

    # The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$) [40 dB]
    # Note that this is $\frac{\beta_{0} P}{\sigma^{2} \Gamma}{=}40 \text{dB}$ as indicated in Matt's manuscript
    REFERENCE_SNR_AT_1_METER = 1e4

    # The propagation environment specific parameter ($z_{1}$) for LoS/NLoS probability determination
    PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61

    # The propagation environment specific parameter ($z_{2}$) for LoS/NLoS probability determination
    PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16

    """
    UAV Motion Model & Power Consumption Profile Parameters
    """

    # The maximum UAV velocity ($V_{\text{max}}$) in m/s
    MAX_UAV_VELOCITY = 55.0

    # The rotor blade tip speed ($U_{\text{tip}}$) in m/s
    ROTOR_BLADE_TIP_SPEED = 200.0

    # The thrust-to-weight ratio in the rotary-wing UAV motion model ($\kappa_{\text{UAV}}{\triangleq}\frac{T}{W}$
    THRUST_TO_WEIGHT_RATIO = 1.0

    # The mean rotor-induced velocity ($v_{0}$) in m/s
    MEAN_ROTOR_INDUCED_VELOCITY = 7.2

    # The UAV power consumption profile constant 1 ($P_{1}$) relevant for blade profile evaluation
    POWER_PROFILE_CONSTANT_1 = 580.65

    # The UAV power consumption profile constant 2 ($P_{2}$) relevant for induced velocity profile evaluation
    POWER_PROFILE_CONSTANT_2 = 790.6715

    # The UAV power consumption profile constant 3 ($P_{3}$) which corresponds to the parasite term
    POWER_PROFILE_CONSTANT_3 = 0.0073

    # The UAV's power consumption while hovering ($P_{\text{mob}}(0)$) in Watts
    UAV_HOVER_POWER_CONSUMPTION = 1371.3215

    # The UAV's power consumption while flying at a power-minimizing velocity of 22 m/s ($P_{\text{mob}}(22)$) in Watts
    UAV_POWER_MINIMIZING_POWER_CONSUMPTION = 936.7679522731312

    """
    Deployment Model Parameters
    """

    # The radius of the circular cell under evaluation ($a$) in meters
    CELL_RADIUS = 1000.0

    # The positional offset to consider during UAV radius level discretization ($a_{\delta}$) in meters
    POSITION_OFFSET = 50.0

    # The height of the UAV from the ground ($H_{U}$) in meters
    UAV_HEIGHT = 200.0

    # The height of the BS from the ground ($H_{B}$) in meters
    BASE_STATION_HEIGHT = 80.0

    # The total number of ground nodes in this deployment
    NUMBER_OF_GROUND_NODES = 30

    # The number of ground nodes per unit area ($\lambda_{G}$) in GNs/m^2
    GROUND_NODE_DENSITY = NUMBER_OF_GROUND_NODES / (math.pi * (CELL_RADIUS ** 2))

    # The number of communication requests originating in the cell per second ($\Lambda$) in requests/second
    ARRIVAL_RATE = 1.67e-2

    # The number of communication requests originating at a GN per second ($\lambda_{P}$) in requests/(GN x second)
    ARRIVAL_RATE_PER_GROUND_NODE = ARRIVAL_RATE / (math.pi * (CELL_RADIUS ** 2) * GROUND_NODE_DENSITY)

    """
    SMDP Design Parameters
    """

    # A waiting state interval: a time period in which no additional request is received, in seconds ($\Delta_{0}$ s)
    WAITING_STATE_INTERVAL = -math.log(0.93) / ARRIVAL_RATE

    # The number of UAV radii "levels" needed for sufficient discretization of the communication state space
    # $N_{\text{sp}}$
    UAV_POSITION_DISCRETIZATION_LEVELS = 25

    # The number of UAV radial velocity "levels" needed for sufficient discretization of the waiting state action space
    # $R_{\text{sp}}$
    UAV_RADIAL_VELOCITY_LEVELS = 25

    # The initial dual variable step-size in the projected sub-gradient ascent algorithm ($\rho_{0}$)
    INITIAL_DUAL_VARIABLE_STEP_SIZE = 1.0

    # The dual variable convergence threshold in the projected sub-gradient ascent algorithm ($\epsilon_{DI}$)
    DUAL_CONVERGENCE_THRESHOLD = 1e-3

    # The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
    BISECTION_METHOD_TOLERANCE = 1e-10

    # The convergence confidence level for optimization algorithms in this framework
    CONVERGENCE_CONFIDENCE = 10

    # The primal feasibility threshold in the projected sub-gradient ascent algorithm ($\epsilon_{PF}$)
    PRIMAL_FEASIBILITY_THRESHOLD = 1e-3

    # The complementary slackness threshold in the projected sub-gradient ascent algorithm ($\epsilon_{CS}$)
    COMPLEMENTARY_SLACKNESS_THRESHOLD = 0.1

    # The termination threshold for the SMDP Value Iteration (VITER) algorithm ($\delta$), i.e.,
    # terminate if $\max_{s{\in}\mathcal{S}} H(s) - \min(x_{s{\in}\mathcal{S}} H(s)) < \delta$
    VITER_TERMINATION_THRESHOLD = 1e-3

    # The learning rate for the angular velocity determination aspect of Lagrangian minimization in the waiting states
    ANGULAR_VELOCITY_LEARNING_RATE = 1e-4

    # The termination threshold for the angular velocity determination aspect of Lagrangian minimization in the
    # waiting states
    ANGULAR_VELOCITY_TERMINATION_THRESHOLD = 1e-10

    """
    The Competitive Swarm Optimization (CSO/HCSO) Design parameters
    """

    # The maximum number of cost function evaluations recommended in the CSO algorithm ($\math{N}_{\text{max}}$)
    MAXIMUM_COST_EVALUATIONS = 1000

    # The initial number of trajectory segments in the HCSO algorithm ($M$)
    INITIAL_TRAJECTORY_SEGMENTS = 6

    # The maximum number of trajectory segments allowed in the HCSO solution ($M_{\text{max}}$)
    MAXIMUM_TRAJECTORY_SEGMENTS = 128

    # A validation multiplier to make sure that the HCSO trajectory parameters are valid vis-à-vis the algorithm
    HCSO_VALIDATION_MULTIPLIER = 10

    # The number of initial trajectory and UAV velocity particles in the HCSO algorithm ($N$) [Swarm Size]
    INITIAL_NUMBER_OF_PARTICLES = 400

    # The number of waypoints to be interpolated between any two given points in the randomly generated $M$-segment
    # trajectories [Random trajectory generation --> Optimization of these trajectories via Interpolation]
    INTERPOLATION_FACTOR = 2

    # The smallest UAV velocity and CSO/HCSO particle velocity value ($V_{\text{low}}$)
    CSO_MINIMUM_VELOCITY_VALUE = 0.0

    # The UAV velocity and CSO/HCSO particle velocity discretization levels
    CSO_VELOCITY_DISCRETIZATION_LEVELS = UAV_RADIAL_VELOCITY_LEVELS

    # The scaling factor employed in the "disturbance around a trajectory reference solution" aspect of HCSO ($\zeta$)
    HCSO_TRAJECTORY_SCALING_FACTOR = 1.0

    # The scaling factor employed in the "disturbance around a velocity reference solution" aspect of HCSO ($\epsilon$)
    HCSO_VELOCITY_SCALING_FACTOR = 1.0

    # The scaling factor that determines the degree of influence of the global means of the trajectory and UAV velocity
    # solutions in the CSO algorithm ($\omega$)
    CSO_PARTICLE_VELOCITY_SCALING_FACTOR = 1.0

    # A namedtuple constituting all the relevant penalty metrics involved in the CSO/HCSO cost function evaluation
    PENALTIES_CAPSULE = namedtuple('penalties_capsule', ['t_p_1', 't_p_2', 'e_p_1', 'e_p_2'])

    """
    Visualization Parameters
    """

    # The Plotly API "markers-only" scatter plot mode
    PLOTLY_MARKERS_MODE = 'markers'

    # The Plotly API "lines and markers" scatter plot mode
    PLOTLY_LINES_MARKERS_MODE = 'lines+markers'

    # The amount of granularity in the UAV positional discretization plot
    UAV_POSITION_DISCRETIZATION_GRANULARITY = 12

    """
    DATA MODEL
    """

    """
    Internal Data Classes
    """

    @dataclass(order=True)
    class SMDPValueIterationDataCapsule:
        """
        The SMDP Value Iteration (VITER) algorithm's internal data capsule (specific to a staged-dual variable value)
        """
        nu: np.float64  # The dual variable ($\nu$)
        o_star: tf.Variable  # The optimal waiting state policy (O^{*})
        u_star: tf.Variable  # The optimal communication state policy (U^{*})
        u_star_indices: tf.Variable  # The optimal action indices w.r.t the comm state policy for visualization
        g: np.float64  # The dual function ($g$)
        e_bar: np.float64  # The average UAV energy consumption for a canary state ($\Bar{E}$)
        t_bar: np.float64  # The average time spent by the UAV per state w.r.t a canary state ($\Bar{T}$)

    @dataclass(order=True)
    class DualMaximizationDataCapsule:
        """
        A dataclass for the information processed during a stage of the projected sub-gradient ascent algorithm
        """
        k: int  # The iteration/stage index ($k$)
        nu_k: np.float64  # The dual variable value in stage-k ($\nu_{k}$)
        g_k: np.float64  # The dual function value in stage-k ($g_{k}$)
        g_k_1: np.float64  # The dual function value in stage-(k-1) ($g_{k-1}$)
        o_star: tf.Variable  # The optimal waiting state policy ($O_{k}^{*}$)
        u_star_k: tf.Variable  # The optimal comm policy ($U_{k}^{*}$)
        e_k: np.float64  # The average energy spent by the UAV in stage-k, w.r.t a canary state ($\Bar{E}_{k}$)
        t_k: np.float64  # The UAV's average time per state, in stage-k, w.r.t a canary state ($\Bar{T}_{k}$)

    """
    ALGORITHMS
    """

    def __init__(self, id__, average_power_constraint, packet_length, evaluators__):
        """
        The initialization sequence: UAV Position Discretization, Ground Node Distribution, UAV Velocity Discretization,
                                     Waiting and Communication States & Actions Initialization

        :param id__: A unique identifier for this evaluator/agent for post-processing visualization
        :param average_power_constraint: The average power consumption constraint for this evaluation ($P_\text{avg}$ W)
        :param packet_length: The packet length ($L$ bits) for all transmissions over all uplink links (GU, UB, GB)
        :param evaluators__: An off-site collection of evaluators/agents to which this instance needs to append itself
                             for post-processing visualization
        """
        self.id = id__
        self.packet_length = packet_length  # The packet length constraint for this "run"
        self.average_power_constraint = average_power_constraint  # The average power constraint for this "run"
        print('[INFO] [{}] MAESTRO Initialization: Bringing things up - Average Power Constraint [P_avg] = '
              '{} kW | Packet Length Constraint [L] = {} Mbits'.format(self.id, (self.average_power_constraint / 1e3),
                                                                       (self.packet_length / 1e6)))
        # Setup with concurrent operations
        with ThreadPoolExecutor(max_workers=256) as executor:
            executor.submit(self.__distribute_ground_nodes)
            executor.submit(self.__discretize_uav_positions, self.UAV_POSITION_DISCRETIZATION_LEVELS,
                            self.UAV_POSITION_DISCRETIZATION_GRANULARITY)
            executor.submit(self.__discretize_uav_radial_velocities, self.UAV_RADIAL_VELOCITY_LEVELS)
            executor.submit(self.__evaluate_power_consumption)  # Power Profile Analyses [Inherent Test Case]
        # State and Action Space encapsulation in "Tensors"
        self.waiting_states = tf.constant(self.uav_positions_polar)
        self.waiting_actions = tf.constant(self.uav_radial_velocity_levels)
        self.comm_states = tf.constant(list(itertools.product(self.uav_positions_rect, self.gn_positions_rect)))
        self.comm_actions = tf.constant(self.uav_positions_rect)
        # Class-wide data collections for visualizations & logging
        self.o_star, self.u_star, self.u_star_indices = None, None, None
        self.relay_status = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.int8)
        self.bs_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.uav_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.comm_delays = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.energy_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.bs_nrg_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.uav_nrg_vals = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.float64)
        self.optimal_trajectories = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], self.comm_actions.shape[0],
                                                                (self.INTERPOLATION_FACTOR *
                                                                 self.MAXIMUM_TRAJECTORY_SEGMENTS), 2],
                                                         dtype=tf.float64), dtype=tf.float64)
        self.optimal_velocities = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], self.comm_actions.shape[0],
                                                              (self.INTERPOLATION_FACTOR *
                                                               self.MAXIMUM_TRAJECTORY_SEGMENTS)],
                                                       dtype=tf.float64), dtype=tf.float64)
        evaluators__.append(self)  # Self-Registration

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    def __distribute_ground_nodes(self):
        """
        Uniformly distribute the GNs in a circular cell of radius $a$ m

        Construct an array of 2-tuples [$(r, \theta)$ in polar coordinates] of GN positions in the circular cell AND
        an array of 2-tuples [$(x, y)$ in rectangular coordinates] of GN positions in the circular cell
        """
        a, lambda_g = self.CELL_RADIUS, self.GROUND_NODE_DENSITY
        number_of_gns = int(lambda_g * math.pi * (a ** 2))
        radii = np.random.uniform(0, a ** 2, number_of_gns) ** 0.5
        angles = np.random.uniform(0, 2 * np.pi, number_of_gns)
        x_coords, y_coords = radii * np.cos(angles), radii * np.sin(angles)
        # Class-wide initialization of positional variables [A concurrency mandated op]
        self.gn_positions_polar, self.gn_positions_rect = list(zip(radii, angles)), list(zip(x_coords, y_coords))

    def __discretize_uav_positions(self, discretization_level, granularity_for_visualization):
        """
        Discretize the UAV positions using equi-spaced radii values in the circular cell

        :param discretization_level: The number of radii levels of UAV position required for state space discretization
        :param granularity_for_visualization: The number of angular variations on each UAV position radius level
        """
        a, offset = self.CELL_RADIUS, self.POSITION_OFFSET
        # Class-wide initialization of a positional variable [A concurrency mandated op]
        self.uav_positions_polar = np.linspace(offset, a - offset, discretization_level, dtype=np.float64)
        angles = np.linspace(0, 2 * np.pi, granularity_for_visualization, dtype=np.float64)
        cosines, sines = np.cos(angles), np.sin(angles)
        coords = np.array([r * np.einsum('ji', np.vstack([cosines, sines])) for r in self.uav_positions_polar])
        x_coords, y_coords = coords[:, :, 0].flatten(), coords[:, :, 1].flatten()
        # Class-wide initialization of a positional variable [A concurrency mandated op]
        self.uav_positions_rect = list(zip(x_coords, y_coords))

    def __evaluate_power_consumption(self, uav_flying_velocity=None):
        """
        Determine the amount of power consumed by the UAV in Watts ($P_{\text{mob}}(V)$ W)

        :param uav_flying_velocity: The instantaneous horizontal flying velocity of the UAV in m/s ($v_{u}(t))

        :return: The UAV's power consumption when flying at the specified velocity (or hovering) according to the
                 provided motion and power consumption profiles

        :raises AssertionError: Assertion failed.
        """
        v, u_tip, v_0 = uav_flying_velocity, self.ROTOR_BLADE_TIP_SPEED, self.MEAN_ROTOR_INDUCED_VELOCITY
        hover_pwr_qa, min_pwr_qa = self.UAV_HOVER_POWER_CONSUMPTION, self.UAV_POWER_MINIMIZING_POWER_CONSUMPTION
        p_1, p_2, p_3 = self.POWER_PROFILE_CONSTANT_1, self.POWER_PROFILE_CONSTANT_2, self.POWER_PROFILE_CONSTANT_3
        # Functional testing call
        if v is None:
            hover_pwr, min_pwr = self.__evaluate_power_consumption(0.0), self.__evaluate_power_consumption(22.0)  # In W
            assert (hover_pwr == hover_pwr_qa) and (min_pwr == min_pwr_qa)  # Test case for the UAV's Power Profile
            return
        # Normal routine call
        return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + \
               (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5) + \
               (p_3 * (v ** 3))

    def __discretize_uav_radial_velocities(self, num_levels):
        """
        Discretize the UAV radial velocity actions, i.e., $O(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$

        :param num_levels: The number of UAV radial velocity levels for discretization of the action space
        """
        v_max = self.MAX_UAV_VELOCITY
        # Class-wide initialization of a motion variable [A concurrency mandated op]
        self.uav_radial_velocity_levels = np.linspace(-v_max, v_max, num_levels, dtype=np.float64)

    def __f_z(self, z):
        """
        Calculate the value of $f(Z)$, i.e., the Shannon-Hartley Channel Capacity in terms of the variable $Z$

        :param z: The optimization variable in the re-formulated primal rate optimization problem

        :return: The value of the function $f(Z)$ in the re-formulated primal rate optimization problem
        """
        b = self.CHANNEL_BANDWIDTH
        return b * np.log2(1 + (0.5 * (z ** 2)))

    # noinspection PyMethodMayBeStatic
    def __marcum_q(self, df, nc, x):
        """
        Calculate the value of the Marcum-Q function using the specified values of the number of degrees of freedom $k$,
        the non-centrality factor $\lambda$, and the random variable rendition $x$

        :param df: The number of degrees of freedom ($k$) for this non-central chi-squared distribution
        :param nc: The non-centrality parameter ($\lambda$) for this non-central chi-squared distribution
        :param x: The random variable rendition ($x$) for CDF evaluation of this non-central chi-squared distribution

        :return: The value of the Marcum-Q function (1 - CDF[non-central chi-squared])
        """
        return 1 - ncx2.cdf(x, df, nc)

    def __f(self, z, *args):
        """
        Calculate the value of the primal objective function in $Z$ based on the provided value of the optimization
        variable ($Z$) and associated args (df, nc, y)

        :param z: The value of the optimization variable to be plugged into the rate adaptation primal objective
                  function, along with its associated args (df, nc, x)
        :param args: The number of degrees of freedom (df), the non-centrality parameter (nc), and an incomplete random
                     variable rendition (y) of the non-central chi-squared distribution used to evaluate the
                     Marcum-Q function

        :return: The value of the rate-adaptation primal objective function after plugging in the provided value of the
                 optimization variable ($Z$) in a certain stage of the bisection method, along with its associated
                 args (df, nc, y)
        """
        df, nc, y = args
        f_z, q_m = self.__f_z(z), self.__marcum_q(df, nc, (y * (z ** 2)))
        ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
        ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf
        return -ln_f_z - ln_q_m

    # noinspection PyMethodMayBeStatic
    def __bisect(self, f, df, nc, y, low, high, tolerance):
        """
        A method to perform bisection-based optimization for rate-adaptation

        :param f: The objective function being optimized
        :param df: The number of degrees of freedom ($k$) for the non-central chi-squared distribution
        :param nc: The non-centrality parameter ($\lambda$) for the non-central chi-squared distribution
        :param y: An incomplete version random variable rendition ($y$) for CDF evaluation of the non-central
                  chi-squared distribution, i.e., $x{=}y Z^{2}$
        :param low: The lower bound of the function's domain
        :param high: The upper bound of the function's domain
        :param tolerance: The tolerance level, which when achieved should terminate the optimization

        :return: The minimizer of the provided objective function, i.e., argmin

        :raises AssertionError: Assertion failed.
        """
        args = (df, nc, y)
        assert tolerance is not None
        mid, converged, conf, conf_th = 0.0, False, 0, self.CONVERGENCE_CONFIDENCE
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
        """
        Calculate the value of the optimization variable $Z$

        :param gamma: The data rate ($\mathbf{\gamma}$) of Tx, employed in the calculation of $Z$

        :return: The value of the optimization variable
                 $Z{=}\sqrt{\frac{2 \beta P}{\sigma^{2} \Gamma} u(\mathbf{\gamma}, \beta)}$
        """
        b = self.CHANNEL_BANDWIDTH
        return np.sqrt(2 * ((2 ** (gamma / b)) - 1))

    def __u(self, gamma, d, los):
        """
        Calculate the value of the function $u(\mathbf{\gamma}, \beta)$

        :param gamma: The data rate ($\mathbf{\gamma}$) of Tx, employed to find $Z$ through $u(\mathbf{\gamma}, \beta)$
        :param d: The distance between the Tx and Rx nodes for determining the large-scale channel variations term
                  $\beta$ for use in the calculation of this function $u(\mathbf{\gamma}, \beta)$
        :param los: This boolean member will be True if the calculation of this function is for LoS settings

        :return: The value of the function $u(\mathbf{\gamma}, \beta)$
        """
        b, gamma_ = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER
        alpha, alpha_, kappa = self.LoS_PATH_LOSS_EXPONENT, self.NLoS_PATH_LOSS_EXPONENT, self.NLoS_ATTENUATION_CONSTANT
        return ((2 ** (gamma / b)) - 1) / (gamma_ * (lambda: kappa, lambda: 1)[los]() *
                                           (d ** (lambda: -alpha_, lambda: -alpha)[los]()))

    def __evaluate_los_throughput(self, d, phi, r_star_los):
        """
        Calculate the contribution of LoS transmissions to the total link throughput

        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                    (GU, UB, GB)
        :param r_star_los: The supplementary optimized LoS throughput member for post-evaluation visualization -- fed
                           into the routine for reference-based concurrent updates
        """
        k_1, k_2 = self.LoS_RICIAN_FACTOR_1, self.LoS_RICIAN_FACTOR_2,
        b, gamma_ = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER
        k, alpha = k_1 * np.exp(k_2 * phi), self.LoS_PATH_LOSS_EXPONENT
        df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), self.BISECTION_METHOD_TOLERANCE
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) *
                                                                            gamma_ * (d ** -alpha))), t)
        gamma_star = self.__f_z(z_star)
        tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)

    def __evaluate_nlos_throughput(self, d, r_star_nlos):
        """
        Calculate the contribution of NLoS transmissions to the total link throughput

        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param r_star_nlos: The supplementary optimized NLoS throughput member for post-evaluation visualization -- fed
                            into the routine for reference-based concurrent updates
        """
        alpha_, t = self.NLoS_PATH_LOSS_EXPONENT, self.BISECTION_METHOD_TOLERANCE
        b, gamma_, kappa = self.CHANNEL_BANDWIDTH, self.REFERENCE_SNR_AT_1_METER, self.NLoS_ATTENUATION_CONSTANT
        df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))
        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa *
                                                    (d ** -alpha_))), t)
        gamma_star = self.__f_z(z_star)
        tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)

    def __calculate_adapted_throughput(self, d, phi, r_star_los, r_star_nlos):
        """
        Calculate the throughput of the link under analysis (GU, UB, GB) (LoS/NLoS) post rate-adaptation in $Z$

        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                    (GU, UB, GB)
        :param r_star_los: The supplementary optimized LoS throughput member for post-evaluation visualization -- fed
                           into the routine for reference-based concurrent updates
        :param r_star_nlos: The supplementary optimized NLoS throughput member for post-evaluation visualization -- fed
                            into the routine for reference-based concurrent updates

        :return: The rate-adapted throughput for the uplink link under analysis (GU, UB, GB) (LoS/NLoS)
        """
        with ThreadPoolExecutor(max_workers=256) as executor:
            executor.submit(self.__evaluate_los_throughput, d, phi, r_star_los)
            executor.submit(self.__evaluate_nlos_throughput, d, r_star_nlos)

    # noinspection PyMethodMayBeStatic
    def __interpolate_waypoints(self, p_indices, p, res_multiplier):
        """
        Increase the resolution of the provided trajectory solution, i.e., increase the number of waypoints
        (called from both HCSO and CSO)

        :param p_indices: The array of indices for referencing the trajectory solution during scipy splining
        :param p: The trajectory solution whose resolution is to be improved
        :param res_multiplier: The number of waypoints in the returned zip object would be 'res_multiplier' times the
                               number of waypoints in the original

        :return: A zip object containing the interpolated x and y coordinates of the trajectory solution
        """
        m = len(p_indices)
        spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)
        return tf.constant(list(zip(spl_x(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)),
                                    spl_y(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64)))))

    # noinspection PyMethodMayBeStatic
    def __interpolate_velocities(self, v_indices, v, res_multiplier):
        """
        Increase the resolution of the provided trajectory solution, i.e., increase the number of waypoints
        (called from both HCSO and CSO)

        :param v_indices: The array of indices for referencing the UAV velocities solution during scipy splining
        :param v: The UAV velocities solution whose resolution is to be improved
        :param res_multiplier: The number of UAV velocities in the returned zip object would be 'res_multiplier' times
                               the number of UAV velocities in the original

        :return: A zip object containing the interpolated velocities from the original UAV velocities solution
        """
        m, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
        return spl_v(np.linspace(0, m - 1, res_multiplier * m, dtype=np.float64))

    def __penalties(self, p__, v__, x_g, res_multiplier):
        """
        Determine the time and energy penalties associated with both phases of the D&F protocol

        :param p__: The M-segment trajectory solution for which the penalty metrics have to be evaluated [Mx1 tensor]
        :param v__: The UAV velocities over the M-segments of the trajectory solution for which the penalty metrics
                    have to be evaluated [Mx1 tensor]
        :param x_g: The coordinates of the Ground Node (GN) originating the communication request ($\mathbf{x}_{G}$)
                    [$[x_{G}, y_{G}]$ numpy array]
        :param res_multiplier: Interpolate the current $\mathbf{p}$ and $\mathbf{v}$ tensors to new sizes
                               current_size * res_multiplier

        :return: A namedtuple constituting all the relevant time and energy penalties
        """
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
        with ThreadPoolExecutor(max_workers=256) as executor:
            for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
                executor.submit(self.__calculate_adapted_throughput, d_gu__, phi_gu__, r_los_gu[i__], r_nlos_gu[i__])
        phi_degrees_gu = (180.0 / np.pi) * phi_gu
        p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
        p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
        r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))
        h_1 = self.packet_length - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))
        t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
        e_p_1 = (lambda: 0.0, lambda: (min_power * t_p_1))[h_1.numpy() > 0.0]()
        # Forward (UAV --> BS)
        r_ub = tf.norm(p[midpoint:], axis=1)
        h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)
        d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
        phi_ub = tf.asin(tf.divide(h_ub, d_ub))
        r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=256) as executor:
            for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
                executor.submit(self.__calculate_adapted_throughput, d_ub__, phi_ub__, r_los_ub[i__], r_nlos_ub[i__])
        phi_degrees_ub = (180.0 / np.pi) * phi_ub
        p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
        p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
        r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))
        h_2 = self.packet_length - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))
        t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
        e_p_2 = (lambda: 0.0, lambda: (min_power * t_p_2))[h_2.numpy() > 0.0]()
        return self.PENALTIES_CAPSULE(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __power_cost(self, v, num_workers=256):
        """
        A decorated routine to enable faster application of the power consumption evaluation method on the
        UAV velocity tensor

        :param v: The UAV flying velocity tensor against which the power consumption evaluation is to be performed
        :param num_workers: The number of parallel runs allowed/considered during this power consumption evaluation on v

        :return: [tf.map_fn] on [evaluate_power_consumption] with [parallel_iterations = num_workers]
        """
        return tf.map_fn(self.__evaluate_power_consumption, v, parallel_iterations=num_workers)

    def __calculate_comm_cost(self, p, v, nu, x_g, f_hat, e_hat=None, t_hat=None):
        """
        The core penalty metric ($\hat{f}_{(\mathbf{p}, \mathbf{v})}$) evaluation routine in the HCSO/CSO algorithms;
        Also used for Lagrangian cost calculations in the outer decision-making framework, i.e., SMDP Value Iteration

        :param p: The trajectory solution ($\mathbf{p}$) for which the penalty metric
                  ($\hat{f}_{(\mathbf{p}, \mathbf{v})}$) is to be evaluated
        :param v: The UAV velocity solution ($\mathbf{v}$) for which the penalty metric
                  ($\hat{f}_{(\mathbf{p}, \mathbf{v})}$) is to be evaluated
        :param nu: The value of the dual variable ($\nu$) in a certain stage of the dual maximization process, for
                   evaluating the penalty metric ($\hat{f}_{(\mathbf{p}, \mathbf{v})}$)
        :param x_g: The coordinates of the Ground Node (GN) originating the communication request ($\mathbf{x}_{G}$)
                    [$[x_{G}, y_{G}]$ numpy array]
        :param f_hat: The cost function value associated with the specified trajectory and UAV velocity solutions
        :param e_hat: The energy consumption due to the UAV's trajectory-specific movements for mobility power analysis
                      in the SMDP Value Iteration Algorithm
        :param t_hat: The time duration of the UAV's trajectory-specific movements for mobility power analysis in the
                      SMDP Value Iteration Algorithm
        """
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

    def __update_winners_and_losers(self, p, v, u, w, t_j, t_j_1, p_bar, v_bar, f_hats, nu, x_g):
        """
        The core pair-wise particle utility evaluation routine of the CSO algorithm

        :param x_g: The coordinates of the Ground Node (GN) originating the communication request ($\mathbf{x}_{G}$)
                    [$[x_{G}, y_{G}]$ numpy array]
        :param p: The NxM matrix of possible trajectory solutions for iterative HCSO-CSO optimization
                  ($\mathbf{p}_{[1:N]}$) [An NxM tensorflow matrix]
        :param v: The NxM matrix of possible UAV velocity solutions for iterative HCSO-CSO optimization
                  ($\mathbf{v}_{[1:N]}$) [An NxM tensorflow matrix]
        :param u: The particle velocities of the provided trajectory solutions ($\mathbf{u}_{[1:N]}$)
                  [An NxM tensorflow matrix]
        :param w: The particle velocities of the provided UAV velocity solutions ($\mathbf{w}_{[1:N]}$)
                  [An NxM tensorflow matrix]
        :param t_j: One constituent index of the pair of particles being evaluated ($t[j]$)
        :param t_j_1: The other constituent index of the pair of particles being evaluated ($t[j+1]$)
        :param p_bar: The global mean of the trajectory solutions at a certain stage of HCSO ($\Bar{\mathbf{p}}$)
        :param v_bar: The global mean of the UAV velocity solutions at a certain stage of HCSO ($\Bar{\mathbf{v}}$)
        :param f_hats: The penalty functions, calculated from $\mathbf{p}_{[1:N]}$ and $\mathbf{v}_{[1:N]}$
                       [An Nx1 tensor]
        :param nu: The value of the dual variable ($\nu$) in a particular iteration of dual maximization
        """
        omega = self.CSO_PARTICLE_VELOCITY_SCALING_FACTOR
        v_min, v_max = self.CSO_MINIMUM_VELOCITY_VALUE, self.MAX_UAV_VELOCITY
        f_hat_t_j, f_hat_t_j_1 = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
        p_t_j, p_t_j_1, v_t_j, v_t_j_1 = p[t_j], p[t_j_1], v[t_j], v[t_j_1]
        u_t_j, u_t_j_1, w_t_j, w_t_j_1 = u[t_j], u[t_j_1], w[t_j], w[t_j_1]
        with ThreadPoolExecutor(max_workers=256) as executor:
            executor.submit(self.__calculate_comm_cost, p_t_j, v_t_j, nu, x_g, f_hat_t_j)
            executor.submit(self.__calculate_comm_cost, p_t_j_1, v_t_j_1, nu, x_g, f_hat_t_j_1)
        argmin__ = np.argmin([f_hat_t_j, f_hat_t_j_1])
        j_win, p_w, v_w, j_los, p_l, v_l, u_l, w_l = (t_j, p_t_j, v_t_j, t_j_1, p_t_j_1, v_t_j_1, u_t_j_1, w_t_j_1) \
            if (argmin__ == 0) else (t_j_1, p_t_j_1, v_t_j_1, t_j, p_t_j, v_t_j, u_t_j, w_t_j)
        r_j = tf.random.uniform(shape=[3, ], dtype=tf.float64)
        # Trajectory particles & associated particle velocities updates
        tf.compat.v1.assign(f_hats[t_j], f_hat_t_j, validate_shape=True, use_locking=True)
        u_j_los = tf.add_n([tf.multiply(r_j[0], u_l), tf.multiply(r_j[1], tf.subtract(p_w, p_l)),
                            omega * tf.multiply(r_j[2], tf.subtract(p_bar, p_l))])
        p_j_los = tf.add(p_l, u_j_los)
        tf.compat.v1.assign(p[j_los], p_j_los, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(u[j_los], u_j_los, validate_shape=True, use_locking=True)
        # UAV velocity particles & associated particle velocities updates
        tf.compat.v1.assign(f_hats[t_j_1], f_hat_t_j_1, validate_shape=True, use_locking=True)
        w_j_los = tf.add_n([tf.multiply(r_j[0], w_l), tf.multiply(r_j[1], tf.subtract(v_w, v_l)),
                            omega * tf.multiply(r_j[2], tf.subtract(v_bar, v_l))])
        v_j_los = tf.clip_by_value(tf.add(v_l, w_j_los), v_min, v_max)
        tf.compat.v1.assign(v[j_los], v_j_los, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(w[j_los], w_j_los, validate_shape=True, use_locking=True)

    def __competitive_swarm_optimization(self, initial_uav_position, terminal_uav_position, ground_node_position,
                                         trajectory_particles, uav_velocity_particles, trajectory_particle_velocities,
                                         uav_velocity_particle_velocities, swarm_size, segment_size, dual_variable):
        """
        The Competitive Swarm Optimization (CSO) Algorithm for UAV trajectory optimization in the communication states

        :param initial_uav_position: The initial UAV position in the form $\mathbf{x}_{0}{=}(r_{U}, 0)$
        :param terminal_uav_position: The final UAV position in the form
                                      $\mathbf{x}_{M}{=}\hat{r}_{U}\frac{\mathbf{x}_{M-1}}{||\mathbf{x}_{M-1}||_{2}}$,
                                      if $||\mathbf{x}_{M-1}||_{2}{\neq}0$, else $\mathbf{x}_{M}{=}(\hat{r}_{U}, 0)$
        :param ground_node_position: The position of the Ground Node (GN) generating the communication request
        :param trajectory_particles: The NxM matrix of possible trajectory solutions for iterative HCSO-CSO
                                     optimization ($\mathbf{p}_{[1:N]}$) [An NxMx2 tensorflow matrix]
        :param uav_velocity_particles: The NxM matrix of possible UAV velocity solutions for iterative HCSO-CSO
                                       optimization ($\mathbf{v}_{[1:N]}$) [An NxM tensorflow matrix]
        :param trajectory_particle_velocities: The particle velocities of the provided trajectory solutions
                                               ($\mathbf{u}_{[1:N]}$) [An NxMx2 tensorflow matrix]
        :param uav_velocity_particle_velocities: The particle velocities of the provided UAV velocity solutions
                                                 ($\mathbf{w}_{[1:N]}$) [An NxM tensorflow matrix]
        :param swarm_size: The number of trajectory & UAV velocity particles passed down from HCSO ($N$)
        :param segment_size: The size of each trajectory & UAV velocity particle passed down from HCSO ($M$)
        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization

        :return: The winning trajectory & UAV velocity particles being returned to HCSO, along with the associated cost
        """
        x_0, x_m, x_g = initial_uav_position, terminal_uav_position, ground_node_position
        p, v = trajectory_particles, uav_velocity_particles
        p_bar, v_bar = tf.reduce_mean(p, axis=0), tf.reduce_mean(v, axis=0)
        u, w = trajectory_particle_velocities, uav_velocity_particle_velocities
        k, n, m, nu, k_max = 0, swarm_size, segment_size, dual_variable, self.MAXIMUM_COST_EVALUATIONS
        f_hats = tf.Variable(tf.zeros(shape=[n, ], dtype=tf.float64), dtype=tf.float64)
        indices = [_ for _ in range(n)]
        while k <= k_max:
            t = tf.random.shuffle(indices)
            with ThreadPoolExecutor(max_workers=256) as executor:
                for j in range(0, n, 2):
                    executor.submit(self.__update_winners_and_losers, p, v, u, w, t[j], t[j + 1],
                                    p_bar, v_bar, f_hats, nu, x_g)
            k += 1
        i = min(indices, key=lambda k__: f_hats[k__].numpy())
        return p[i], u[i], v[i], w[i], f_hats[i]

    # noinspection PyUnusedLocal
    def __hierarchical_competitive_swarm_optimization(self, state_index, action_index, initial_uav_position,
                                                      terminal_uav_position, dual_variable, ground_node_position,
                                                      lagrangian, energy, time_duration):
        """
        The Hierarchical Competitive Swarm Optimization (HCSO) Algorithm to find the optimal trajectory and UAV velocity
        solutions ($(\mathbf{p}^{*}, \mathbf{v}^{*})$), and their associated Lagrangian cost metric value which is
        fed into the SMDP Value Iteration (VITER) Algorithm

        :param state_index: The state index into the lagrangians tensor for cost metric update post-evaluation
                            [concurrency requirement]
        :param action_index: The action index into the lagrangians tensor for cost metric update post-evaluation
                             [concurrency requirement]
        :param initial_uav_position: The initial UAV position in the form $\mathbf{x}_{0}{=}(r_{U}, 0)$
        :param terminal_uav_position: The final UAV position in the form
                                      $\mathbf{x}_{M}{=}\hat{r}_{U}\frac{\mathbf{x}_{M-1}}{||\mathbf{x}_{M-1}||_{2}}$,
                                      if $||\mathbf{x}_{M-1}||_{2}{\neq}0$, else $\mathbf{x}_{M}{=}(\hat{r}_{U}, 0)$
        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization
        :param ground_node_position: The coordinates of the Ground Node (GN) originating the communication request
                                     ($\mathbf{x}_{G}$)
        :param lagrangian: The Lagrangian cost metric for a value update associated with the fed-in comm
                            state and action [concurrency requirement]
        :param energy: The energy cost metric for a value update associated with the fed-in comm
                         state and action [concurrency requirement]
        :param time_duration: The time-duration cost metric for a value update associated with the fed-in
                               comm state and action [concurrency requirement]
        """
        i__, j__ = state_index, action_index
        p_star, u_star, v_star, w_star = None, None, None, None
        i, nu, f_hat = 0, dual_variable, tf.Variable(0.0, dtype=tf.float64)
        opt_traj, opt_velo = self.optimal_trajectories, self.optimal_velocities
        l_comm_star, e_comm_star, t_comm_star = lagrangian, energy, time_duration
        xi_s, delays, nrgs = self.relay_status, self.comm_delays, self.energy_vals
        e_usage, delta = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
        z_1, z_2 = self.PROPAGATION_ENVIRONMENT_PARAMETER_1, self.PROPAGATION_ENVIRONMENT_PARAMETER_2
        a, v_levels, p_len = self.CELL_RADIUS, self.CSO_VELOCITY_DISCRETIZATION_LEVELS, self.packet_length
        v_min, v_max, eps = self.CSO_MINIMUM_VELOCITY_VALUE, self.MAX_UAV_VELOCITY, self.HCSO_VELOCITY_SCALING_FACTOR
        n, m_old, m_ip = self.INITIAL_NUMBER_OF_PARTICLES, self.INITIAL_TRAJECTORY_SEGMENTS, self.INTERPOLATION_FACTOR
        bs_delays, bs_nrgs, uav_delays, uav_nrgs = self.bs_delays, self.bs_nrg_vals, self.uav_delays, self.uav_nrg_vals
        x_0, x_m, x_g, h_b = initial_uav_position, terminal_uav_position, ground_node_position, self.BASE_STATION_HEIGHT
        zeta, m, m_max = self.HCSO_TRAJECTORY_SCALING_FACTOR, m_old, self.MAXIMUM_TRAJECTORY_SEGMENTS
        assert m_max > m and m_max >= m * self.HCSO_VALIDATION_MULTIPLIER

        r_gen = RandomTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
        d_gen = DeterministicTrajectoriesGeneration(x_0, x_m, (-a, a), (0, 2 * np.pi), n, m_old, m_ip)
        p, m = tf.concat([d_gen.generate_optimize(), r_gen.optimize(r_gen.generate())], axis=0), ((m + 2) * m_ip)

        velocity_vals = np.linspace(v_min, v_max, v_levels)
        u = tf.Variable(np.random.choice(velocity_vals, size=[n, m, 2]))
        v = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))
        w = tf.Variable(np.random.choice(velocity_vals, size=[n, m]))
        # HCSO while loop
        while m <= m_max * m_ip:
            p_star, u_star, v_star, w_star, f_star = self.__competitive_swarm_optimization(x_0, x_m, x_g, p, v, u, w,
                                                                                           n, m, nu)
            n -= 20 * (i + 1)
            i += 1
            indices = [_ for _ in range(m)]
            # Trajectory particles
            p_tilde = tf.tile(tf.expand_dims(self.__interpolate_waypoints(indices, p_star, m_ip), axis=0),
                              multiples=[n, 1, 1])
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
            tf.compat.v1.assign(p[:, :-1, :], tf.add(p_tilde[:, :-1, :], p_tilde_r), validate_shape=True,
                                use_locking=True)
            tf.compat.v1.assign(p[:, -1, :], tf.add(p_tilde[:, -1, :],
                                                    tf.random.normal(shape=[scale_last.shape[0], 2],
                                                                     mean=tf.zeros(shape=[scale_last.shape[0], 2],
                                                                                   dtype=tf.float64),
                                                                     stddev=tf.tile(tf.sqrt(scale_last),
                                                                                    multiples=[1, 2]),
                                                                     dtype=tf.float64)),
                                validate_shape=True, use_locking=True)
            tf.compat.v1.assign(u, tf.tile(tf.expand_dims(self.__interpolate_waypoints(indices, u_star, m_ip), axis=0),
                                           multiples=[n, 1, 1]), validate_shape=True, use_locking=True)

            # UAV velocity particles
            shape_v = [n, m_ip * m]
            v_tilde = tf.tile(tf.expand_dims(self.__interpolate_velocities(indices, v_star, m_ip), axis=0),
                              multiples=[n, 1])
            v = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
            w = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
            v_tilde_r = tf.random.normal(shape=shape_v, mean=tf.zeros(shape=shape_v, dtype=tf.float64),
                                         stddev=tf.sqrt(tf.multiply(eps * ((v_max - v_min) ** 2),
                                                                    tf.ones(shape=shape_v, dtype=tf.float64))),
                                         dtype=tf.float64)
            tf.compat.v1.assign(v, tf.clip_by_value(tf.add(v_tilde, v_tilde_r), v_min, v_max),
                                validate_shape=True, use_locking=True)
            tf.compat.v1.assign(w, tf.tile(tf.expand_dims(self.__interpolate_velocities(indices, w_star, m_ip), axis=0),
                                           multiples=[n, 1]), validate_shape=False, use_locking=True)
            m *= m_ip
        # Lagrangian cost determination for scheduling (Direct BS or UAV relay?)
        d_gb = np.sqrt(np.add(np.square(h_b), np.square(np.linalg.norm(x_g))))
        phi_gb = np.arcsin(tf.divide(h_b, d_gb))
        r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=256) as executor:
            executor.submit(self.__calculate_comm_cost, p_star, v_star, nu, x_g, f_hat, e_usage, delta)
            executor.submit(self.__calculate_adapted_throughput, d_gb, phi_gb, r_los_gb, r_nlos_gb)
        phi_degrees_gb = (180.0 / np.pi) * phi_gb
        p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
        p_nlos_gb = 1 - p_los_gb
        r_bar_gb = tf.add(tf.multiply(p_los_gb, r_los_gb), tf.multiply(p_nlos_gb, r_nlos_gb))
        l_xi_0 = (p_len / r_bar_gb) if r_bar_gb != 0.0 else np.inf
        l_xi_1, e_xi_1, t_xi_1 = f_hat.numpy(), e_usage.numpy(), delta.numpy()
        l__, e__, t__, xi__ = (l_xi_1, e_xi_1, t_xi_1, 1) if (l_xi_1 < l_xi_0) else (l_xi_0, 0.0, l_xi_0, 0)
        # Tensor updates for further processing
        tf.compat.v1.assign(l_comm_star[i__, j__], l__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(e_comm_star[i__, j__], e__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(t_comm_star[i__, j__], t__, validate_shape=True, use_locking=True)
        # Class-wide data collection updates for visualization
        tf.compat.v1.assign(xi_s[i__], xi__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(nrgs[i__], e__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(delays[i__], t__, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(bs_nrgs[i__], 0.0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(uav_nrgs[i__], e_xi_1, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(bs_delays[i__], l_xi_0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(uav_delays[i__], t_xi_1, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(opt_traj[i__, j__], p_star, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(opt_velo[i__, j__], v_star, validate_shape=True, use_locking=True)

    def __angular_velocity_optimization(self, state_index, action_index, uav_position, radial_velocity,
                                        dual_variable, lagrangians, energies, time_durations):
        """
        Optimize the angular velocity term, and determine the associated Lagrangian, energy, and temporal costs for the
        specified waiting state ($s{\in}\mathcal{S}_{\text{wait}}$) and radial velocity action ($v_{r}$)

        :param state_index: The state index into the lagrangians, energies, and time-durations tensor for cost metric
                            update post-evaluation [concurrency requirement]
        :param action_index: The action index into the lagrangians, energies, and time-durations tensor for cost metric
                             update post-evaluation [concurrency requirement]
        :param uav_position: The waiting state constituting the UAV position in polar coordinates
                             (just the radius level: $r_{U}$)
        :param radial_velocity: The waiting state action constituting the radial velocity value ($v_{r}$)
        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization
        :param lagrangians: The Lagrangian cost metrics collection for a value update associated with the fed-in
                            waiting state and action [concurrency requirement]
        :param energies: The energy cost metrics collection for a value update associated with the fed-in waiting
                         state and action [concurrency requirement]
        :param time_durations: The time-duration cost metrics collection for a value update associated with the fed-in
                               waiting state and action [concurrency requirement]
        """
        l_wait_star, e_wait_star, t_wait_star = lagrangians, energies, time_durations
        theta_c, delta_c = np.random.random(), self.ANGULAR_VELOCITY_TERMINATION_THRESHOLD
        id__, p_avg, delta_0 = self.id, self.average_power_constraint, self.WAITING_STATE_INTERVAL
        i__, j__, r_u, v_r, nu = state_index, action_index, uav_position.numpy(), radial_velocity.numpy(), dual_variable
        v_max, flying_velocity = self.MAX_UAV_VELOCITY, lambda x: ((v_r ** 2) + ((r_u ** 2) * (x ** 2))) ** 0.5

        def objective(x): return nu * (self.__evaluate_power_consumption(flying_velocity(x)) - p_avg) * delta_0

        constraints = ({'type': 'ineq', 'fun': lambda x: v_max - flying_velocity(x)})
        theta_c_star = minimize(objective, theta_c, method='SLSQP', constraints=constraints, tol=delta_c).x[0]
        p_mob_star = self.__evaluate_power_consumption(flying_velocity(theta_c_star))
        tf.compat.v1.assign(l_wait_star[i__, j__], nu * (p_mob_star - p_avg) * delta_0,
                            validate_shape=True, use_locking=True)
        tf.compat.v1.assign(e_wait_star[i__, j__], p_mob_star * delta_0, validate_shape=True, use_locking=True)
        tf.compat.v1.assign(t_wait_star[i__, j__], delta_0, validate_shape=True, use_locking=True)

    def __optimize_waiting_states(self, nu, s_wait, a_wait, l_wait_star, e_wait_star, t_wait_star):
        """
        Optimize the waiting states of the UAV, i.e., for each waiting state ($s{\in}\mathcal{S}_{\text{wait}}$),
        determine the optimal radial velocity action of the UAV ($v_{r}$)

        :param nu: The value of the dual variable ($\nu$) that has been fed in through the Projected Sub-gradient
                   Ascent Algorithm
        :param s_wait: The waiting state space (tensor) of the UAV ($\mathcal{S}_{\text{wait}}$)
        :param a_wait: The waiting action space (tensor) of the UAV ($\mathcal{A}_{\text{wait}}$)
        :param l_wait_star: The Lagrangian costs (final optimality) collection (tensor) for waiting states and actions
                            ($l_{\nu}^{*}(s; v_{r})$)
        :param e_wait_star: The energy costs (final optimality) collection (tensor) for waiting states and actions
                            ($e_{\nu}^{*}(s; v_{r}, \theta_{c}^{*})$)
        :param t_wait_star: The temporal costs (final optimality) collection (tensor) for waiting states and actions
                            ($t_{\nu}^{*}(s; v_{r}, \theta_{c}^{*})$)
        """
        print('[DEBUG] [{}] MAESTRO __optimize_waiting_states: Waiting states optimization '
              'started...'.format(self.id))
        try:
            with ThreadPoolExecutor(max_workers=256) as executor:
                for i__, r_u in enumerate(s_wait):
                    for j__, v_r in enumerate(a_wait):
                        executor.submit(self.__angular_velocity_optimization, i__, j__, r_u, v_r, nu,
                                        l_wait_star, e_wait_star, t_wait_star)
            print('[DEBUG] [{}] MAESTRO __optimize_waiting_states: Waiting states optimization '
                  'completed...'.format(self.id))
        except Exception as e__:
            print('[ERROR] [{}] MAESTRO __optimize_waiting_states: Exception caught during waiting '
                  'state optimization - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))

    def __optimize_comm_states(self, nu, s_comm, a_comm, l_comm_star, e_comm_star, t_comm_star):
        """
        Optimize the comm states of the UAV, i.e., for each comm state ($s{\in}\mathcal{S}_{\text{comm}}$),
        determine the optimal position of the UAV ($\hat{r}_{U}$ or $\hat{\mathbf{x}}_{U}$ or $(x_{U}, y_{U})$)

        :param nu: The value of the dual variable ($\nu$) that has been fed in through the Projected Sub-gradient
                   Ascent Algorithm
        :param s_comm: The comm state space (tensor) of the UAV ($\mathcal{S}_{\text{comm}}$)
        :param a_comm: The comm action space (tensor) of the UAV ($\mathcal{A}_{\text{comm}}$)
        :param l_comm_star: The Lagrangian costs (final optimality) collection (tensor) for comm states and actions
                            ($l_{\nu}^{*}(s; \hat{\mathbf{x}}_{U}, 0 or 1)$)
        :param e_comm_star: The energy costs (final optimality) collection (tensor) for comm states and actions
                            ($e_{\nu}^{*}(s; \hat{\mathbf{x}}_{U}, 0 or 1)$)
        :param t_comm_star: The temporal costs (final optimality) collection (tensor) for comm states and actions
                            ($t_{\nu}^{*}(s; \hat{\mathbf{x}}_{U}, 0 or 1)$)
        """
        print('[DEBUG] [{}] MAESTRO __optimize_comm_states: Comm states optimization started...'.format(self.id))
        try:
            with ThreadPoolExecutor(max_workers=256) as executor:
                for i__, s in enumerate(s_comm):
                    for j__, x_u in enumerate(a_comm):
                        x_0 = tf.constant([s[0].numpy()], dtype=tf.float64)
                        x_u = tf.constant([x_u.numpy()], dtype=tf.float64)
                        x_g = tf.constant([s[1].numpy()], dtype=tf.float64)
                        executor.submit(self.__hierarchical_competitive_swarm_optimization,
                                        i__, j__, x_0, x_u, nu, x_g, l_comm_star, e_comm_star, t_comm_star)
                print('[DEBUG] [{}] MAESTRO __optimize_comm_states: Comm states optimization '
                      'completed...'.format(self.id))
        except Exception as e__:
            print('[ERROR] [{}] MAESTRO __optimize_comm_states: Exception caught during comm '
                  'state optimization - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))

    def __smdp_waiting_viter_updates(self, l_wait_star, e_wait_star, t_wait_star, v_i_wait, v_i_comm, h_i_wait,
                                     e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait, o_star):
        """
        The updates to the value functions of waiting states are performed in this routine, along with similar updates
        to the energy consumption and time-duration costs of these waiting states

        :param l_wait_star: The Lagrangian costs (contains optimalities) collection (tensor) for waiting states and
                            actions ($l_{\nu}^{*}(s; v_{r})$)
        :param e_wait_star: The energy costs (contains optimalities) collection (tensor) for waiting states and actions
                            ($e_{\nu}^{*}(s; v_{r})$)
        :param t_wait_star: The temporal costs (contains optimalities) collection (tensor) for waiting states and
                            actions ($t_{\nu}^{*}(s; v_{r})$)
        :param v_i_wait: The value function collection (tensor to be updated here) for waiting states and actions
                         ($V(s)$): Updated using SMDP State Transitions and Comm State Value Functions
        :param v_i_comm: The value function collection (tensor to be updated here) for comm states and actions
                         ($V(s)$)
        :param h_i_wait: The value function difference evaluator collection in order to determine the termination
                       status of the SMDP Value Iteration Algorithm
        :param e_i_wait: The value iteration staged UAV energy consumption collection for waiting states
                         ($E_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$)
        :param e_i_comm: The value iteration staged UAV energy consumption collection for comm states
                         ($E_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{comm}}$)
        :param t_i_wait: The value iteration staged time duration collection for waiting states
                         ($T_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$)
        :param t_i_comm: The value iteration staged time duration collection for comm states
                         ($T_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{comm}}$)
        :param s_wait: The UAV waiting states collection ($\mathcal{S}_{\text{wait}}$)
        :param s_comm: The UAV comm states collection ($\mathcal{S}_{\text{comm}}$)
        :param a_wait: The UAV wait state actions ($\mathcal{A}_{\text{wait}}$)
        :param o_star: The final optimal waiting state policy ($O^{*}(s)$)
        """
        print('[DEBUG] [{}] MAESTRO __smdp_waiting_viter_updates: SMDP VITER updates started...'.format(self.id))
        try:
            lambda__, delta_0 = self.ARRIVAL_RATE, self.WAITING_STATE_INTERVAL
            v_i_1_wait = tf.constant(v_i_wait)
            s_wait_added = tf.add(tf.tile(tf.expand_dims(s_wait, axis=1), multiples=[1, a_wait.shape[0]]),
                                  delta_0 * a_wait)
            wait_indices = tf.argmin(tf.abs(tf.subtract(tf.tile(tf.expand_dims(s_wait_added, axis=2),
                                                                multiples=[1, 1, s_wait.shape[0]]), s_wait)), axis=2)
            # Waiting state value functions: Transition to another waiting state
            v_i_wait_remapped = tf.gather(tf.tile(tf.expand_dims(v_i_wait, axis=1), multiples=[1, a_wait.shape[0]]),
                                          wait_indices)[:, :, 0]
            # Comm state value functions: Transition to a comm state
            mins_org = tf.reshape(tf.gather(s_wait, wait_indices), shape=[-1])
            v_i_comm_remapped = tf.reshape(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(v_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org), axis=1),
                shape=[s_wait.shape[0], a_wait.shape[0]])
            _v_i_added = tf.add(l_wait_star, math.exp(-lambda__ * delta_0) * v_i_wait_remapped,
                                (1 - math.exp(-lambda__ * delta_0)) * v_i_comm_remapped)
            o_i_1 = tf.argmin(_v_i_added, axis=1)
            o_star__ = tf.gather(a_wait, o_i_1)
            tf.compat.v1.assign(v_i_wait, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(h_i_wait, tf.subtract(v_i_wait, v_i_1_wait), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(o_star, o_star__, validate_shape=True, use_locking=True)
            s_wait_added_min = tf.add(s_wait, delta_0 * o_star__)
            wait_indices_min = tf.squeeze(tf.argmin(tf.abs(tf.subtract(tf.tile(tf.expand_dims(s_wait_added_min, axis=1),
                                                                               multiples=[1, s_wait.shape[0]]),
                                                                       s_wait)), axis=1))
            # Waiting state energy & time costs
            e_wait_star_remapped = tf.boolean_mask(e_wait_star, tf.one_hot(o_i_1, e_wait_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            e_i_wait_remapped = tf.gather(e_i_wait, wait_indices_min)
            t_wait_star_remapped = tf.boolean_mask(t_wait_star, tf.one_hot(o_i_1, t_wait_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            t_i_wait_remapped = tf.gather(t_i_wait, wait_indices_min)
            # Comm state energy & time costs
            mins_org_min = tf.gather(s_wait, wait_indices_min)
            e_i_comm_remapped = tf.squeeze(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(e_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org_min),
                axis=1))
            t_i_comm_remapped = tf.squeeze(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(t_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org_min),
                axis=1))
            # Energy cost
            _e_i_added = tf.add(e_wait_star_remapped, math.exp(-lambda__ * delta_0) * e_i_wait_remapped,
                                (1 - math.exp(-lambda__ * delta_0)) * e_i_comm_remapped)
            tf.compat.v1.assign(e_i_wait, _e_i_added, validate_shape=True, use_locking=True)
            # Delay (time duration) cost
            _t_i_added = tf.add(t_wait_star_remapped, math.exp(-lambda__ * delta_0) * t_i_wait_remapped,
                                (1 - math.exp(-lambda__ * delta_0)) * t_i_comm_remapped)
            tf.compat.v1.assign(t_i_wait, _t_i_added, validate_shape=True, use_locking=True)
            print('[DEBUG] [{}] MAESTRO __smdp_waiting_viter_updates: SMDP VITER updates '
                  'completed...'.format(self.id))
        except Exception as e__:
            print(
                '[ERROR] [{}] MAESTRO __smdp_waiting_viter_updates: Exception caught during SMDP VITER waiting '
                'state updates - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))

    def __smdp_comm_viter_updates(self, l_comm_star, e_comm_star, t_comm_star, v_i_comm, v_i_wait, h_i_comm,
                                  e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm, u_star, u_star_indices):
        """
        The updates to the value functions of comm states are performed in this routine, along with similar updates
        to the energy consumption and time-duration costs of these comm states

        :param l_comm_star: The Lagrangian costs (contains optimalities) collection (tensor) for comm states and
                            actions ($l_{\nu}^{*}(s; \hat{r}_{U})$)
        :param e_comm_star: The energy costs (contains optimalities) collection (tensor) for comm states and actions
                            ($e_{\nu}^{*}(s; \hat{r}_{U})$)
        :param t_comm_star: The temporal costs (contains optimalities) collection (tensor) for comm states and
                            actions ($t_{\nu}^{*}(s; \hat{r}_{U})$)
        :param v_i_comm: The value function collection (tensor to be updated here) for comm states and actions
                         ($V(s)$): Updated using SMDP State Transitions and Waiting State Value Functions
        :param v_i_wait: The value function collection (tensor to be updated here) for waiting states and actions
                         ($V(s)$)
        :param h_i_comm: The value function difference evaluator collection in order to determine the termination
                       status of the SMDP Value Iteration Algorithm
        :param e_i_comm: The value iteration staged UAV energy consumption collection for comm states
                         ($E_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{comm}}$)
        :param e_i_wait: The value iteration staged UAV energy consumption collection for waiting states
                         ($E_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$)
        :param t_i_comm: The value iteration staged time duration collection for comm states
                         ($T_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{comm}}$)
        :param t_i_wait: The value iteration staged time duration collection for waiting states
                         ($T_{i}(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$)
        :param s_wait: The UAV waiting states collection ($\mathcal{S}_{\text{wait}}$)
        :param a_comm: The UAV comm state actions ($\mathcal{A}_{\text{comm}}$)
        :param u_star: The final optimal waiting state policy ($U^{*}(s)$)
        :param u_star_indices: The optimal action indices for visualization processing
        """
        print('[DEBUG] [{}] MAESTRO __smdp_comm_viter_updates: SMDP VITER updates started...'.format(self.id))
        try:
            v_i_1_comm = tf.constant(v_i_comm)
            _v_i_wait = tf.tile(tf.expand_dims(tf.squeeze(tf.map_fn(lambda x: tf.gather(v_i_wait,
                                                                                        tf.where(tf.equal(s_wait, x))),
                                                                    tf.norm(a_comm, axis=1))), axis=0),
                                multiples=[l_comm_star.shape[0], 1])
            _v_i_added = tf.add(l_comm_star, _v_i_wait)  # Transition only to the waiting state
            u_i_1 = tf.argmin(_v_i_added, axis=1)
            u_star__ = tf.gather(a_comm, u_i_1, axis=0)
            tf.compat.v1.assign(u_star_indices, u_i_1, validate_shape=True, use_locking=True)
            tf.compat.v1.assign(v_i_comm, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(h_i_comm, tf.subtract(v_i_comm, v_i_1_comm), validate_shape=True, use_locking=True)
            tf.compat.v1.assign(u_star, u_star__, validate_shape=True, use_locking=True)
            # Comm state energy costs
            e_i_wait_remapped = tf.squeeze(tf.map_fn(lambda x: tf.gather(e_i_wait, tf.where(tf.equal(s_wait, x))),
                                                     tf.norm(u_star__, axis=1)))
            e_comm_star_remapped = tf.boolean_mask(e_comm_star, tf.one_hot(u_i_1, e_comm_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            tf.compat.v1.assign(e_i_comm, tf.add(e_comm_star_remapped, e_i_wait_remapped),
                                validate_shape=True, use_locking=True)
            # Comm state time duration costs
            t_i_wait_remapped = tf.squeeze(tf.map_fn(lambda x: tf.gather(t_i_wait, tf.where(tf.equal(s_wait, x))),
                                                     tf.norm(u_star__, axis=1)))
            t_comm_star_remapped = tf.boolean_mask(t_comm_star, tf.one_hot(u_i_1, t_comm_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            tf.compat.v1.assign(t_i_comm, tf.add(t_comm_star_remapped, t_i_wait_remapped),
                                validate_shape=True, use_locking=True)
            print('[DEBUG] [{}] MAESTRO __smdp_comm_viter_updates: '
                  'SMDP VITER updates completed...'.format(self.id))
        except Exception as e__:
            print('[ERROR] [{}] MAESTRO __smdp_comm_viter_updates: Exception caught during SMDP VITER comm '
                  'state updates - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))

    def __pi_comm(self):
        """
        A simple routine to evaluate the steady state probability of being in the communication state
        ($\pi_{\text{comm}}{=}\frac{1 - e^{-\Lambda \Delta_{0}}}{2 - e^{-\Lambda \Delta_{0}}})

        :return: The steady state probability of being in the communication state ($\pi_{\text{comm}}$)
        """
        lambda_, delta_0 = self.ARRIVAL_RATE, self.WAITING_STATE_INTERVAL
        p_e = math.exp(-lambda_ * delta_0)
        return (1 - p_e) / (2 - p_e)

    def __value_iteration(self, dual_variable):
        """
        The SMDP Value Iteration Algorithm for optimal waiting state & communication state policy determination, given
        a particular value of the dual variable for Lagrangian cost metric evaluation

        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization

        :return: The SMDP_VITER_DATA_CAPSULE namedtuple constituting the following members:
                 $(O_{k}^{*}, U_{k}^{*}, g_{k}, \Bar{E}_{k}, \Bar{T}_{k})$
        """
        print('[DEBUG] [{}] MAESTRO __value_iteration: SMDP Value Iteration started...'.format(self.id))
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
            with ThreadPoolExecutor(max_workers=256) as executor:
                executor.submit(self.__optimize_waiting_states, nu, s_wait, a_wait,
                                l_wait_star, e_wait_star, t_wait_star)
                executor.submit(self.__optimize_comm_states, nu, s_comm, a_comm,
                                l_comm_star, e_comm_star, t_comm_star)
            # Value function updates (via SMDP state transitions: one step future look-ahead)
            converged, conf, conf_th = False, 0, self.CONVERGENCE_CONFIDENCE
            while not converged or conf < conf_th:
                with ThreadPoolExecutor(max_workers=256) as executor:
                    executor.submit(self.__smdp_waiting_viter_updates, l_wait_star, e_wait_star, t_wait_star, v_i_wait,
                                    v_i_comm, h_i_wait, e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait,
                                    o_star)
                    executor.submit(self.__smdp_comm_viter_updates, l_comm_star, e_comm_star, t_comm_star, v_i_comm,
                                    v_i_wait, h_i_comm, e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm,
                                    u_star, u_star_indices)
                converged = (tf.reduce_max(h_i_wait) - tf.reduce_min(h_i_wait) < delta) and \
                            (tf.reduce_max(h_i_comm) - tf.reduce_min(h_i_comm) < delta)
                conf += 1 if converged else -conf  # Update OR Reset if it is a red herring
                i += 1
            pi_comm = self.__pi_comm()
            divisor = 1 / (pi_comm * i) if pi_comm != 0.0 else np.inf
            g = divisor * v_i_comm[canary_index].numpy()
            e_bar = divisor * e_i_comm[canary_index].numpy()
            t_bar = divisor * t_i_comm[canary_index].numpy()
            print('[DEBUG] [{}] MAESTRO __value_iteration: SMDP Value Iteration completed...'.format(self.id))
            return self.SMDPValueIterationDataCapsule(nu=nu, o_star=o_star, u_star=u_star,
                                                      u_star_indices=u_star_indices, g=g, e_bar=e_bar, t_bar=t_bar)
        except Exception as e__:
            print('[ERROR] [{}] MAESTRO __value_iteration: Exception caught during value iteration '
                  'updates - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))
        return None

    def projected_subgradient_ascent(self):
        """
        The Projected Sub-gradient Ascent Algorithm for dual variable maximization
        """
        print('[DEBUG] [{}] MAESTRO projected_subgradient_ascent: Projected Sub-Gradient Ascent '
              'started...'.format(self.id))
        th_d, th_pf = self.DUAL_CONVERGENCE_THRESHOLD, self.PRIMAL_FEASIBILITY_THRESHOLD
        k, g_k_1, p_av, rho_0 = 0, 0.0, self.average_power_constraint, self.INITIAL_DUAL_VARIABLE_STEP_SIZE
        th_cs, converged, conf, conf_th = self.COMPLEMENTARY_SLACKNESS_THRESHOLD, False, 0, self.CONVERGENCE_CONFIDENCE
        try:
            nu_k, o_star_k, u_star_k, u_star_indices_k, g_k, e_k, t_k = astuple(self.__value_iteration(0.01))
            while not converged or conf < conf_th:
                print('[DEBUG] [{}] MAESTRO projected_subgradient_ascent: A new iteration of the '
                      'Projected Sub-Gradient Ascent method has been started...'.format(self.id))
                g_k_1, nu_k = g_k, max(nu_k + ((rho_0 / (k + 1)) * (e_k - (p_av * t_k))), 0)
                nu_k, o_star_k, u_star_k, u_star_indices_k, g_k, e_k, t_k = astuple(self.__value_iteration(nu_k))
                converged = (abs(g_k - g_k_1) < th_d) and (e_k - (p_av * t_k) < th_pf) and \
                            (nu_k * abs(e_k - (p_av * t_k)) < th_cs)
                conf += 1 if converged else -conf  # Update OR Reset if it is a red herring
                k += 1
            self.o_star, self.u_star, self.u_star_indices = o_star_k, u_star_k, u_star_indices_k
            print('[DEBUG] [{}] MAESTRO projected_subgradient_ascent: Projected Sub-Gradient Ascent '
                  'completed | Logging outputs...'.format(self.id))
            # noinspection PyTypeChecker
            log_outputs(self.id, self.packet_length, self.average_power_constraint, self.waiting_states,
                        self.waiting_actions, self.comm_states, self.comm_actions,
                        self.comm_delays, self.energy_vals,
                        self.bs_delays, self.bs_nrg_vals, self.uav_delays, self.uav_nrg_vals,
                        self.optimal_trajectories, self.optimal_velocities, self.relay_status, self.o_star, self.u_star)
        except Exception as e__:
            print('[ERROR] [{}] MAESTRO projected_subgradient_ascent: Exception caught during projected '
                  'sub-gradient ascent updates - {}'.format(self.id, traceback.print_tb(e__.__traceback__)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        if exc_tb is not None:
            print(
                '[ERROR] [{}] MAESTRO Termination: Tearing things down - Exception Type = {} | '
                'Exception Value = {} | Traceback = {}'.format(self.id, exc_type, exc_val, traceback.print_tb(exc_tb)))


"""
External Handling Ops
"""


def launch_evaluation(id_, power_constraint, packet_len_constraint, evaluators_):
    """
    Launch an SMDPEvaluation instance given the constraints in order to analyze its performance and functionalities

    :param id_: A unique identifier for the evaluator/agent that is to be launched from this routine
    :param power_constraint: The UAV average power consumption constraint for an SMDPEvaluation instance
    :param packet_len_constraint: The data packet length constraint for an SMDPEvaluation instance
    :param evaluators_: A collection to house the evaluator instances for post-processing
    """
    with MAESTRO(id_, power_constraint, packet_len_constraint, evaluators_) as evaluator__:
        evaluator__.projected_subgradient_ascent()


def log_outputs(identifier, packet_length, avg_power_constraint, wait_states, wait_actions, comm_states, comm_actions,
                comm_delays, energy_values, bs_delays, bs_energies, uav_delays, uav_energies, optimal_trajs,
                optimal_velos, relay_statuses, optimal_wait_policy, optimal_comm_policy):
    """
    Log the core outputs and performance metrics in JSON format for each evaluator identifier by a UUID

    :param identifier: A unique identifier for the evaluator/agent whose outputs & metrics are to be logged
    :param packet_length: The data payload size constraint for the provided SMDPEvaluation evaluator
    :param avg_power_constraint: The UAV average power consumption constraint for the provided SMDPEvaluation evaluator
    :param wait_states: A collection of waiting states employed in the evaluation of the provided SMDPEvaluation agent
    :param wait_actions: A collection of waiting actions employed in the evaluation of the provided SMDPEvaluation agent
    :param comm_states: A collection of comm states employed in the evaluation of the provided SMDPEvaluation agent
    :param comm_actions: A collection of comm actions employed in the evaluation of the provided SMDPEvaluation agent
    :param comm_delays: The optimal communication service delays associated with each comm state post policy convergence
    :param energy_values: The optimal energy consumption values associated with each comm state post policy convergence
    :param bs_delays: The BS-driven communication service delays associated with each comm state post-convergence
    :param bs_energies: The BS-driven energy consumption values associated with each comm state post-convergence
    :param uav_delays: The UAV-driven communication service delays associated with each comm state post-convergence
    :param uav_energies: The UAV-driven energy consumption values associated with each comm state post-convergence
    :param optimal_trajs: A collection of optimal trajectories corresponding to the comm states in the SMDP model
    :param optimal_velos: A collection of optimal UAV velocities corresponding to the comm states in the SMDP model
    :param relay_statuses: A collection of relay statuses corresponding to the comm states in the SMDP model
    :param optimal_wait_policy: The optimal waiting state policy that is to be logged for off-site visualizations
    :param optimal_comm_policy: The optimal communication state policy that is to be logged for off-site visualizations
    """
    # ID | L | P_avg | S_wait | A_wait | S_comm | A_comm | O_star | U_star | xi_star | p_star | delays | energies
    tf.io.write_file(str(identifier), tf.strings.format('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\{}',
                                                        (tf.constant(str(identifier), dtype=tf.string),
                                                         tf.constant(str(packet_length), dtype=tf.string),
                                                         tf.constant(str(avg_power_constraint), dtype=tf.string),
                                                         tf.constant(str(wait_states.numpy()), dtype=tf.string),
                                                         tf.constant(str(wait_actions.numpy()), dtype=tf.string),
                                                         tf.constant(str(comm_states.numpy()), dtype=tf.string),
                                                         tf.constant(str(comm_actions.numpy()), dtype=tf.string),
                                                         tf.constant(str(comm_delays.numpy()), dtype=tf.string),
                                                         tf.constant(str(energy_values.numpy()), dtype=tf.string),
                                                         tf.constant(str(bs_delays.numpy()), dtype=tf.string),
                                                         tf.constant(str(bs_energies.numpy()), dtype=tf.string),
                                                         tf.constant(str(uav_delays.numpy()), dtype=tf.string),
                                                         tf.constant(str(uav_energies.numpy()), dtype=tf.string),
                                                         tf.constant(str(optimal_trajs.numpy()), dtype=tf.string),
                                                         tf.constant(str(optimal_velos.numpy()), dtype=tf.string),
                                                         tf.constant(str(relay_statuses.numpy()), dtype=tf.string),
                                                         tf.constant(str(optimal_wait_policy.numpy()), dtype=tf.string),
                                                         tf.constant(str(optimal_comm_policy.numpy()), dtype=tf.string)
                                                         )), name='logging_outputs')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] [Main Thread] MAESTRO main: Starting the evaluation of the proposed SMDP-HCSO formulation '
          'for adaptive multi-scale scheduling and trajectory optimization of power-constrained UAV relays...')
    evaluators = list()  # A collection to house the evaluator instances for post-processing
    # Test samples
    avg_powers, packet_lens = np.array([1e3]), np.array([1e6])
    with ThreadPoolExecutor(max_workers=256) as exxeggutor:
        for packet_len in packet_lens:
            for avg_power in avg_powers:
                exxeggutor.submit(launch_evaluation, uuid.uuid4(), avg_power, packet_len, evaluators)
# The evaluation ends here...
