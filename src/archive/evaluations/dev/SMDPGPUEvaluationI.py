"""
This script encapsulates the following algorithms employed in the adaptive scheduling and trajectory optimization of
multi-winged, power-constrained, rotary UAVs:
    1. The value iteration algorithm for the SMDP formulation,
    2. The projected sub-gradient ascent algorithm for dual variable maximization,
    3. The Hierarchical Competitive Swarm Optimization (HCSO) algorithm for optimal trajectory determination, and
    4. The Competitive Swarm Optimization (CSO) algorithm for lower-level trajectory optimization (driven by HCSO).

The plots involved in this evaluation are:
    1. UAV Radial Velocity ($v_{r}$ m/s) v UAV Radius Level ($r_{U}$ m) w.r.t the optimal waiting state policy, for
       L = 1.0, 5.0, and 10.0 Mbits;
    2. Expected Average Delay ($\Bar{T}$ s) v UAV Average Power Constraint ($P_{\text{avg}}$ W) w.r.t the optimal
       policy, for L = 1.0, 5.0, and 10.0 Mbits;
    3. Map of GNs ($x$ m v $y$ m) that transmit directly to the BS w.r.t the optimal policy, for L = 1.0, 5.0, and
       10.0 Mbits;
    4. The UAV trajectories ($x$ m v $y$ m) for $3$ randomly chosen GNs w.r.t the optimal policy; and
    5. The convergence of the Lagrangian cost metrics in the comm states ($l_{\nu}^{*}(s;\hat{r}_{U}, 1)$ v Time [s]).

Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
Version v21.03: Mathematical Modeling by Matthew Bliss <mbliss@purdue.edu> and Nicolo Michelusi <michelus@purdue.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

# The Multi-Machine Multi-GPU Variant

import os

# Logging setup for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The imports
import time
import math
import uuid
import plotly
import warnings
import itertools
import traceback
import numpy as np
import tensorflow as tf
from threading import Lock
from collections import namedtuple
import plotly.graph_objs as graph_objs
from dataclasses import dataclass, astuple
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor
import tensorflow_constrained_optimization as tfco
from plotly.validators.scatter.marker import SymbolValidator

"""
Plotly API settings & instantiations
"""

plotly.tools.set_credentials_file(username='<User_Name>', api_key='<API_Key>')  # User Credentials
raw_symbols = [val for i, val in enumerate(SymbolValidator().values) if (i % 2) == 1]  # Allowed marker symbols

# noinspection SpellCheckingInspection
colors = '''
    aliceblue, antiquewhite, aqua, aquamarine, azure,
    beige, bisque, black, blanchedalmond, blue,
    blueviolet, brown, burlywood, cadetblue,
    chartreuse, chocolate, coral, cornflowerblue,
    cornsilk, crimson, cyan, darkblue, darkcyan,
    darkgoldenrod, darkgray, darkgrey, darkgreen,
    darkkhaki, darkmagenta, darkolivegreen, darkorange,
    darkorchid, darkred, darksalmon, darkseagreen,
    darkslateblue, darkslategray, darkslategrey,
    darkturquoise, darkviolet, deeppink, deepskyblue,
    dimgray, dimgrey, dodgerblue, firebrick,
    floralwhite, forestgreen, fuchsia, gainsboro,
    ghostwhite, gold, goldenrod, gray, grey, green,
    greenyellow, honeydew, hotpink, indianred, indigo,
    ivory, khaki, lavender, lavenderblush, lawngreen,
    lemonchiffon, lightblue, lightcoral, lightcyan,
    lightgoldenrodyellow, lightgray, lightgrey,
    lightgreen, lightpink, lightsalmon, lightseagreen,
    lightskyblue, lightslategray, lightslategrey,
    lightsteelblue, lightyellow, lime, limegreen,
    linen, magenta, maroon, mediumaquamarine,
    mediumblue, mediumorchid, mediumpurple,
    mediumseagreen, mediumslateblue, mediumspringgreen,
    mediumturquoise, mediumvioletred, midnightblue,
    mintcream, mistyrose, moccasin, navajowhite, navy,
    oldlace, olive, olivedrab, orange, orangered,
    orchid, palegoldenrod, palegreen, paleturquoise,
    palevioletred, papayawhip, peachpuff, peru, pink,
    plum, powderblue, purple, red, rosybrown,
    royalblue, saddlebrown, salmon, sandybrown,
    seagreen, seashell, sienna, silver, skyblue,
    slateblue, slategray, slategrey, snow, springgreen,
    steelblue, tan, teal, thistle, tomato, turquoise,
    violet, wheat, white, whitesmoke, yellow,
    yellowgreen
    '''
raw_colors = colors.split(',')
raw_colors = [c.replace('\n', '') for c in raw_colors]
raw_colors = [c.replace(' ', '') for c in raw_colors]  # Allowed marker/line colors (CSS colors in Plotly)

"""
TensorFlow library settings & instantiations
"""

# TensorFlow GPU evaluation
gpu_details = tf.config.list_physical_devices('GPU')
print('[INFO] [Main Thread] TensorFlow settings_test: Number of GPUs available on this machine = [{}] | '
      'GPU Information = [{}]'.format(len(gpu_details), gpu_details))

# Enable device placement logging to see which devices the ops and tensors are assigned to...
tf.debugging.set_log_device_placement(True)

# Cluster & Scope definitions
# cluster_spec = tf.train.ClusterSpec({'worker': ['localhost:2222', 'localhost:2223',
#                                                 'localhost:2224', 'localhost:2225']})
# simple_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, task_type='worker', task_id=0)
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=simple_resolver)

strategy = tf.distribute.MirroredStrategy()

"""
Miscellaneous Global Settings
"""

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# numpy.random seed
np.random.seed(777)

# A global resource access semaphore for concurrent evaluations and analysis ops on shared visualization data objects
lock = Lock()

"""
Internal Optimization Problem Class
"""


# Core Descriptive Internal Class
class UAVAngularVelocityOptimization(tfco.ConstrainedMinimizationProblem):
    """
    This class describes the optimization problem associated with minimizing the Lagrangian metric w.r.t a provided
    waiting state (UAV position radius level - $r_{U}$) and waiting state action (UAV radial velocity - $v_{r}$)

    Derived from the Convex Optimization Example in the TensorFlow Constrained Optimization Library...

    Use the tensorflow-constrained-optimization library by downloading it as follows:
    python -m pip install git+https://github.com/google-research/tensorflow_constrained_optimization
    """

    def __init__(self, caller_id, wrapper, dual_variable, uav_radius_level, uav_radial_velocity_component,
                 uav_angular_velocity_component):
        """
        The initialization sequence for this class: Declarations, Definitions, and Instantiations

        :param caller_id: The identifier (uuid string) associated with the instance that instantiated this problem
        :param wrapper: The callee class reference for cost function & constraint internals
        :param dual_variable: The value of the dual variable ($\nu$) in a specific stage of the Projected Sub-gradient
                              Algorithm
        :param uav_radius_level: The waiting state of the UAV which constitutes the current radius level of the UAV
                                 ($r_{U}$)
        :param uav_radial_velocity_component: The waiting state action of the UAV which constitutes the radial velocity
                                              of the UAV ($v_{r}$)
        :param uav_angular_velocity_component: The angular velocity component of the UAV velocity that is to optimized
                                               in order to minimize the Lagrangian cost metric associated with this
                                               waiting state and waiting state action ($\theta_{c}$)
        """
        super().__init__()
        self.caller_id = caller_id
        print('[INFO] [{}] UAVAngularVelocityOptimization Initialization: Bringing things up...'.format(self.caller_id))

        self.wrapper = wrapper
        self.dual_variable = dual_variable
        self.uav_radius_level = uav_radius_level
        self.uav_radial_velocity = uav_radial_velocity_component
        self.uav_angular_velocity_component = uav_angular_velocity_component

        # The initialization sequence has been completed

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    @property
    def num_constraints(self):
        """
        Get the number of constraints involved in this optimization [API requirement]

        :return The number of constraints involved in this optimization [1: max velocity constraint]
        """
        return 1

    def objective(self):
        """
        The Lagrangian cost metric evaluation function ($l_{nu}^{*}(s; v_{r})$)

        :return: The objective function (Lagrangian evaluation) that is to be optimized
        """
        theta_c = self.uav_angular_velocity_component
        nu, r_u, v_r = self.dual_variable, self.uav_radius_level, self.uav_radial_velocity
        p_avg, delta_0 = self.wrapper.average_power_constraint, self.wrapper.WAITING_STATE_INTERVAL
        return nu * (self.wrapper.mobility_pwr(((v_r ** 2) + ((r_u ** 2) * (theta_c ** 2))) ** 0.5) -
                     p_avg) * delta_0

    def constraints(self):
        """
        Define the constraints associated with this optimization problem

        :return: A tensor with the constraints involved in this optimization problem [max velocity constraint]
        """
        theta_c = self.uav_angular_velocity_component
        r_u, v_r, v_max = self.uav_radius_level, self.uav_radial_velocity, self.wrapper.MAX_UAV_VELOCITY
        return tf.stack([((((v_r ** 2) + ((r_u ** 2) * (theta_c ** 2))) ** 0.5) - v_max)])

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the script that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        print('[INFO] [{}] UAVAngularVelocityOptimization Termination: Tearing things down - Exception Type = {} | '
              'Exception Value = {} | Traceback = {}'.format(self.caller_id, exc_type, exc_val, exc_tb))

        # Nothing to do here...


"""
Internal Random Optimized Trajectories Generation Class 
"""


# Core Descriptive Internal Class
class RandomTrajectoriesGeneration(object):
    """
    This class encapsulates the operations associated with generating a set of trajectories (size = swarm_size $N$)
    between two given points (in rectangular coordinates: $\mathbf{x}_{0}$ and $\mathbf{x}_{M}$).
    """

    def __init__(self, caller_id, source, destination, radial_bounds, angular_bounds, swarm_size,
                 segment_size, interpolation_num):
        """
        The initialization sequence

        v21.03: Circular Cell of radius $a$ meters | 2-dimensional random walks | Optimization via Interpolation

        :param caller_id: The identifier (uuid string) associated with the instance that instantiated this generator
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
        self.caller_id = caller_id
        print(f'[INFO] [{self.caller_id}] RandomTrajectoriesGeneration Initialization: Bringing things up...')

        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num

        # self.strategy = tf.distribute.MirroredStrategy()

        # The initialization sequence has been completed

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    def __generate_segment(self, traj):
        """
        Generate a random $M$-segment trajectory

        :param traj: Assign the generated random $M$-segment trajectory to this [indexed] collection [Mx2 tensor]
        """
        (r_min, r_max), (th_min, th_max) = self.r_bounds, self.theta_bounds
        r = tf.random.uniform(shape=[self.m, ], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[self.m, ], minval=th_min, maxval=th_max, dtype=tf.float64)
        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)),
                                             tf.multiply(r, tf.math.sin(theta))], axis=1),
                            validate_shape=True, use_locking=False)

        # Nothing to return...

    # Core routine
    def generate(self):
        """
        Generate $N$ (self.n) random $M$-segment (self.m) trajectories between $\mathbf{x}_{0}$ (self.x_0) and
        $\mathbf{x}_{M}$ (self.x_m) [Concurrent Ops | Pre-Optimization]

        :return: An (N, M, 2) tensor: $N$ $M$-trajectory segments with each point represented as a 2-tuple (x, y) in
                 rectangular coordinates
        """
        n, m = self.n, self.m
        # The random trajectories' collection initialization
        trajs = tf.Variable(tf.zeros(shape=(n, m, 2), dtype=tf.float64), dtype=tf.float64)

        # Concurrent generation of random trajectories pre-optimization
        # with self.strategy.scope():
        with ThreadPoolExecutor(max_workers=(2000 * n)) as executor:
            for i in range(n):
                executor.submit(self.__generate_segment, trajs[i, :])

        return trajs

    def __optimize_segment(self, traj, opt_traj):
        """
        Optimize the randomness for the given $M$-segment trajectory via Scipy UnivariateSpline interpolation

        :param traj: The [indexed] random trajectory generated via uniform sampling in polar coordinates [Mx2 tensor]
        :param opt_traj: The [indexed] new trajectory collection to house the interpolated one [Concurrency Op]
                         [(m_ip * (M + 2))x2 tensor]
        """
        i_s = [_ for _ in range(traj.shape[0] + 2)]  # +2 to account for the concat of source and destination points
        a, x_0, x_m, m_pre, m_ip = self.r_bounds[1], self.x_0, self.x_m, len(i_s), self.m_ip
        x = np.linspace(0, (m_pre - 1), (m_ip * m_pre), dtype=np.float64)
        tf.compat.v1.assign(opt_traj, tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
                            validate_shape=True, use_locking=False)

        # Nothing to return...

    def optimize(self, trajs):
        """
        Optimize the randomness via Scipy UnivariateSpline interpolation

        :param trajs: The indexed random trajectories generated via uniform sampling in polar coordinates [NxMx2 tensor]

        :return The new trajectories' collection housing the optimized ones [Nx(m_ip * (M + 2))x2 tensor]
                [Concurrency Op]
        """
        n, m_post = self.n, (self.m_ip * (self.m + 2))
        # The optimized trajectories' collection to house the results of the interpolation phase
        opt_trajs = tf.Variable(tf.zeros(shape=[n, m_post, 2], dtype=tf.float64), dtype=tf.float64)

        # Concurrent optimization via interpolation of the generated random trajectories
        # with self.strategy.scope():
        with ThreadPoolExecutor(max_workers=(2000 * n)) as executor:
            for i in range(n):
                executor.submit(self.__optimize_segment, trajs[i, :], opt_trajs[i, :])

        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        print(f'[INFO] [{self.caller_id}] RandomTrajectoriesGeneration Termination: Tearing things down - '
              f'Exception Type = {exc_type} | Exception Value = {exc_val} | Traceback = {exc_tb}')

        # Nothing to do here...


"""
Core class
"""


class SMDPEvaluation(object):
    """
    This class constitutes the optimal scheduling & trajectory determination for a multi-winged, power-constrained,
    rotary Unmanned Aerial Vehicle (UAV), in settings with a Base Station (BS) at the center of a circular cell--dotted
    with a myriad of uniformly-distributed, heterogeneous, Ground Nodes (GNs) generating communication requests
    according to a Poisson process.
    """

    """
    SYSTEM MODEL
    """

    """
    Channel Model Parameters
    """

    # The bandwidth of each orthogonal channel assigned to this application ($B$) in Hz
    # Note that there are $k$ orthogonal channels employed by the BS-UAV combination.
    CHANNEL_BANDWIDTH = 1e6

    # The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
    LoS_PATH_LOSS_EXPONENT = 2.0

    # The propagation environment dependent coefficient ($k_{1}$) for the LoS Rician link model's $K$-factor
    LoS_RICIAN_FACTOR_1 = 1.0

    # The propagation environment dependent coefficient ($k_{2}$) for the LoS Rician link model's $K$-factor
    LoS_RICIAN_FACTOR_2 = 0.0512

    # The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
    NLoS_PATH_LOSS_EXPONENT = 2.8

    # The additional NLoS attenuation factor ($\kappa$)
    NLoS_ATTENUATION_CONSTANT = 0.2

    # The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$) [40 dB]
    REFERENCE_SNR_AT_1_METER = 10e4

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
    # UAV_HOVER_POWER_CONSUMPTION = 1370

    # The UAV's power consumption while flying at a power-minimizing velocity of 22m/s ($P_{\text{mob}}(22)$ W) in Watts
    # UAV_POWER_MINIMIZING_POWER_CONSUMPTION = 940

    """
    Deployment Model Parameters
    """

    # The radius of the circular cell under evaluation ($a$) in meters
    CELL_RADIUS = 100.0

    # The height of the UAV from the ground ($H_{U}$) in meters
    UAV_HEIGHT = 50.0

    # The height of the BS from the ground ($H_{B}$) in meters
    BASE_STATION_HEIGHT = 40.0

    # The number of ground nodes per unit area ($\lambda_{G}$) in GNs/m^2
    GROUND_NODE_DENSITY = 3.2e-4

    # The number of communication requests originating in the cell per second ($\Lambda$) in requests/second
    ARRIVAL_RATE = 0.0167

    # The number of communication requests originating at a GN per second ($\lambda_{P}$) in requests/(GN x second)
    ARRIVAL_RATE_PER_GROUND_NODE = 1.67e-3

    """
    SMDP Design Parameters
    """

    # A waiting state interval: a time period in which no additional request is received, in seconds ($\Delta_{0}$ s)
    WAITING_STATE_INTERVAL = 4.3456

    # The number of UAV radii "levels" needed for sufficient discretization of the communication state space
    UAV_POSITION_DISCRETIZATION_LEVEL = 9

    # The number of UAV radial velocity "levels" needed for sufficient discretization of the waiting state action space
    UAV_RADIAL_VELOCITY_LEVELS = 9

    # The initial dual variable step-size in the projected sub-gradient ascent algorithm ($\rho_{0}$)
    INITIAL_DUAL_VARIABLE_STEP_SIZE = 1.0

    # The dual variable convergence threshold in the projected sub-gradient ascent algorithm ($\epsilon_{DI}$)
    DUAL_CONVERGENCE_THRESHOLD = 1e-3

    # The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
    BISECTION_METHOD_TOLERANCE = 1e-3

    # The convergence confidence level for optimization algorithms in this framework (e.g., for angular velocity opt)
    CONVERGENCE_CONFIDENCE = 3

    # The primal feasibility threshold in the projected sub-gradient ascent algorithm ($\epsilon_{PF}$)
    PRIMAL_FEASIBILITY_THRESHOLD = 1e-3

    # The complementary slackness threshold in the projected sub-gradient ascent algorithm ($\epsilon_{CS}$)
    COMPLEMENTARY_SLACKNESS_THRESHOLD = 1e-3

    # The termination threshold for the SMDP Value Iteration (VITER) algorithm ($\delta$), i.e.,
    #   Terminate if $\max_{s{\in}\mathcal{S}} H(s) - \min(x_{s{\in}\mathcal{S}} H(s)) < \delta$
    VITER_TERMINATION_THRESHOLD = 1e-3

    # The learning rate for the angular velocity determination aspect of Lagrangian minimization in the waiting states
    ANGULAR_VELOCITY_LEARNING_RATE = 0.1

    # The termination threshold for the angular velocity determination aspect of Lagrangian minimization in the
    #   waiting states
    ANGULAR_VELOCITY_TERMINATION_THRESHOLD = 1e-3

    """
    The Competitive Swarm Optimization (CSO/HCSO) Design parameters
    """

    # The maximum number of cost function evaluations recommended in the CSO algorithm ($\math{N}_{\text{max}}$)
    MAXIMUM_COST_EVALUATIONS = 1e4

    # The initial number of trajectory segments in the HCSO algorithm ($M$)
    # Note here that the actual initial interpolated $M$ would be ((14 + 2) * INTERPOLATION_FACTOR)--with the +2
    #   accounting for the initial & final UAV positions
    INITIAL_TRAJECTORY_SEGMENTS = 6

    # The maximum number of trajectory segments allowed in the HCSO solution ($M_{\text{max}}$)
    MAXIMUM_TRAJECTORY_SEGMENTS = 32

    # The number of initial trajectory and UAV velocity particles in the HCSO algorithm ($N$) [Swarm Size]
    INITIAL_NUMBER_OF_PARTICLES = 32

    # The number of waypoints to be interpolated between any two given points in the randomly generated $M$-segment
    #   trajectories [Random trajectory generation --> Optimization of these trajectories via interpolation]
    INTERPOLATION_FACTOR = 2

    # The smallest UAV velocity and CSO/HCSO particle velocity value ($V_{\text{low}}$)
    CSO_MINIMUM_VELOCITY_VALUE = 0.0

    # The UAV velocity and CSO/HCSO particle velocity discretization levels
    CSO_VELOCITY_DISCRETIZATION_LEVELS = 10

    # The scaling factor employed in the "disturbance around a trajectory reference solution" aspect of HCSO ($\zeta$)
    HCSO_TRAJECTORY_SCALING_FACTOR = 0.45

    # The scaling factor employed in the "disturbance around a velocity reference solution" aspect of HCSO ($\epsilon$)
    HCSO_VELOCITY_SCALING_FACTOR = 0.45

    # The scaling factor that determines the degree of influence of the global means of the trajectory and UAV velocity
    #   solutions in the CSO algorithm ($\omega$)
    CSO_PARTICLE_VELOCITY_SCALING_FACTOR = 0.90

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
    UAV_POSITION_DISCRETIZATION_GRANULARITY = 10

    # The number of GNs to consider for UAV trajectory visualization post-convergence
    NUMBER_OF_GROUND_NODES_FOR_TRAJECTORY_VISUALIZATION = 3

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

    def __init__(self, id__, average_power_constraint, packet_length, canary_index, evaluators__):
        """
        The initialization sequence: UAV Position Discretization, Ground Node Distribution, UAV Velocity Discretization,
                                     Waiting and Communication States & Actions Initialization

        :param id__: A unique identifier for this evaluator/agent for post-processing visualization
        :param average_power_constraint: The average power consumption constraint for this evaluation ($P_\text{avg}$ W)
        :param packet_length: The packet length ($L$ bits) for all transmissions over all uplink links (GU, UB, GB)
        :param canary_index: The index of the canary state ("Canary in a coal mine") for visualizations and associated
                             evaluations (w.r.t varying packet_lens, avg_powers, and other factors)
        :param evaluators__: An off-site collection of evaluators/agents to which this instance needs to append itself
                             for post-processing visualization
        """

        # self.lock = Lock()  # The resource access semaphore

        self.id = id__
        self.packet_length = packet_length  # The packet length constraint for this "run"
        self.average_power_constraint = average_power_constraint  # The average power constraint for this "run"

        print('[INFO] [{}] SMDPEvaluation Initialization: Bringing things up - Average Power Constraint [P_avg] = '
              '{} kW | Packet Length Constraint [L] = {} Mbits'.format(self.id, (self.average_power_constraint / 1e3),
                                                                       (self.packet_length / 1e6)))

        # Setup with concurrent operations
        with ThreadPoolExecutor(max_workers=20000) as executor:
            executor.submit(self.__distribute_ground_nodes)
            executor.submit(self.__discretize_uav_positions, self.UAV_POSITION_DISCRETIZATION_LEVEL,
                            self.UAV_POSITION_DISCRETIZATION_GRANULARITY)
            executor.submit(self.__discretize_uav_radial_velocities, self.UAV_RADIAL_VELOCITY_LEVELS)
            executor.submit(self.evaluate_power_consumption)  # Power Profile Analyses [Inherent Test Case]

        # Device placement: Waiting state & action space declarations | Comm state & action space declarations
        with strategy.scope():
            # State and Action Space encapsulation in "Tensors"
            self.waiting_states = tf.constant(self.uav_positions_polar)
            self.waiting_actions = tf.constant(self.uav_radial_velocity_levels)
            self.comm_states = tf.constant(list(itertools.product(self.uav_positions_rect, self.gn_positions_rect)))
            self.comm_actions = tf.constant(self.uav_positions_rect)

            # Class-wide data collections for visualizations
            self.canary_state_index, self.canary_state_comm_costs = canary_index, dict()
            self.o_star, self.u_star, self.canary_comm_average_delay = None, None, 0.0
            self.relay_status = tf.Variable(tf.zeros(shape=[self.comm_states.shape[0], ], dtype=tf.int8), dtype=tf.int8)
            self.optimal_trajectories = tf.Variable(
                tf.zeros(shape=[self.comm_states.shape[0], self.comm_actions.shape[0],
                                (self.INTERPOLATION_FACTOR *
                                 self.MAXIMUM_TRAJECTORY_SEGMENTS), 2
                                ], dtype=tf.float64), dtype=tf.float64)

        # Ex-situ definition of tf.map_fn lambdas for faster processing
        # argmin = lambda x: np.abs(s_wait.numpy() - x).argmin()  # noqa
        # state_update = lambda y: tf.add(y, (delta_0 * a_wait))  # noqa

        evaluators__.append(self)  # Self-Registration

        # The initialization sequence has been completed...

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    # Auxiliary Descriptive Routine
    def __distribute_ground_nodes(self):
        """
        Uniformly distribute the GNs in a circular cell of radius 'a'

        Construct An array of 2-tuples [$(r, \theta)$ in polar coordinates] of GN positions in the circular cell AND
        An array of 2-tuples [$(x, y)$ in rectangular coordinates] of GN positions in the circular cell
        """
        number_of_gns = int(self.GROUND_NODE_DENSITY * np.pi * (self.CELL_RADIUS ** 2))
        radii = np.random.uniform(0, self.CELL_RADIUS ** 2, number_of_gns) ** 0.5
        angles = np.random.uniform(0, 2 * np.pi, number_of_gns)
        x_coords, y_coords = radii * np.cos(angles), radii * np.sin(angles)

        # Plotly plot of Ground Node distribution
        plot_data = [graph_objs.Scatter(x=x_coords, y=y_coords, mode=self.PLOTLY_MARKERS_MODE)]
        plot_layout = dict(title='Distribution of the Ground Nodes in a circular cell of radius '
                                 '{} [Cartesian Coordinate System]'.format(self.CELL_RADIUS),
                           xaxis=dict(title='x'), yaxis=dict(title='y'))
        plot_figure = dict(data=plot_data, layout=plot_layout)
        plot_figure_url = plotly.plotly.plot(plot_figure, filename='Odin_GN_Distribution', auto_open=False)

        print('[DEBUG] [{}] SMDPEvaluation distribute_ground_nodes: A plot representing the distribution of the '
              'Ground Nodes in a circular cell of radius a = {} m has been rendered, and is available here - '
              '{}'.format(self.id, self.CELL_RADIUS, plot_figure_url))

        # Class-wide initialization of positional variables [A concurrency mandated op]
        self.gn_positions_polar, self.gn_positions_rect = list(zip(radii, angles)), list(zip(x_coords, y_coords))

        # Nothing to return...

    # Auxiliary Descriptive Routine
    def __discretize_uav_positions(self, discretization_level, granularity_for_visualization):
        """
        Discretize the UAV positions using equi-spaced radii values in the circular cell

        :param discretization_level: The number of radii levels of UAV positioning required state space discretization
        :param granularity_for_visualization: The number of boundary points on each radius level for visualization
        """
        # Class-wide initialization of a positional variable [A concurrency mandated op]
        self.uav_positions_polar = np.linspace(0, self.CELL_RADIUS, discretization_level, dtype=np.float64)

        angles = np.linspace(0, 2 * np.pi, granularity_for_visualization, dtype=np.float64)
        cosines, sines = np.cos(angles), np.sin(angles)
        coords = np.array([r * np.einsum('ji', np.vstack([cosines, sines])) for r in self.uav_positions_polar])
        x_coords, y_coords = coords[:, :, 0].flatten(), coords[:, :, 1].flatten()

        # Class-wide initialization of a positional variable [A concurrency mandated op]
        self.uav_positions_rect = list(zip(x_coords, y_coords))

        # Plotly plot of UAV position discretization
        plot_data = graph_objs.Scatter(x=x_coords, y=y_coords, mode=self.PLOTLY_MARKERS_MODE)
        plot_layout = dict(title='UAV Position Discretization Levels in a circular cell of radius '
                                 '{} [Cartesian Coordinate System]'.format(self.CELL_RADIUS),
                           xaxis=dict(title='x'), yaxis=dict(title='y'))
        plot_figure = dict(data=[plot_data], layout=plot_layout)
        plot_figure_url = plotly.plotly.plot(plot_figure, filename='Odin_UAV_Position_Discretization', auto_open=False)

        print('[DEBUG] [{}] SMDPEvaluation discretize_uav_positions: A plot representing the discretization of UAV '
              'positions in a circular cell of radius a = {} m has been rendered, and is available here - '
              '{}'.format(self.id, self.CELL_RADIUS, plot_figure_url))

        # Nothing to return

    # Auxiliary Descriptive Routine
    def evaluate_power_consumption(self, uav_flying_velocity=None):
        """
        Determine the amount of power consumed by the UAV in Watts ($P_{\text{mob}}(V)$ W)

        :param uav_flying_velocity: The instantaneous horizontal flying velocity of the UAV in m/s ($v_{u}(t))

        :return: The UAV's power consumption when flying at the specified velocity (or hovering) according to the
                 provided motion and power consumption profiles

        :raises AssertionError: Assertion failed.
        """
        # Functional testing call
        if uav_flying_velocity is None:
            hover_pwr, min_pwr = self.evaluate_power_consumption(0.0), self.evaluate_power_consumption(22.0)  # In W
            assert (hover_pwr == 1371.3215) and (min_pwr == 936.7679522731312)  # Test case for the UAV's Power Profile
            print('[DEBUG] [{}] SMDPEvaluation evaluate_power_consumption: UAV Power Consumption Profile and '
                  'UAV Motion Model Analysis - UAV power consumption while hovering = {} kW | UAV power consumption '
                  'while flying at a power-minimizing forward velocity of 22 m/s = '
                  '{} kW'.format(self.id, (hover_pwr / 1e3), (min_pwr / 1e3)))
            return

        # Normal routine call
        return (self.POWER_PROFILE_CONSTANT_1 * (1 + ((3 * (uav_flying_velocity ** 2)) /
                                                      (self.ROTOR_BLADE_TIP_SPEED ** 2)))) + \
            (self.POWER_PROFILE_CONSTANT_2 * ((((1 + ((uav_flying_velocity ** 4) /
                                                      (4 * (self.MEAN_ROTOR_INDUCED_VELOCITY ** 4)))) ** 0.5) -
                                               ((uav_flying_velocity ** 2) /
                                                (2 * (self.MEAN_ROTOR_INDUCED_VELOCITY ** 2))))) ** 0.5) + \
            (self.POWER_PROFILE_CONSTANT_3 * (uav_flying_velocity ** 3))

    # Auxiliary Descriptive Routine
    def __discretize_uav_radial_velocities(self, num_levels):
        """
        Discretize the UAV radial velocity actions, i.e., $O(s),{\forall}s{\in}\mathcal{S}_{\text{wait}}$

        :param num_levels: The number of UAV radial velocity levels for discretization of the action space
        """
        # Class-wide initialization of a motion variable [A concurrency mandated op]
        self.uav_radial_velocity_levels = np.linspace(-1 * self.MAX_UAV_VELOCITY, self.MAX_UAV_VELOCITY, num_levels,
                                                      dtype=np.float64)

        print('[DEBUG] [{}] SMDPEvaluation __discretize_uav_radial_velocities: The discretized radial velocity levels, '
              'with V_max = {} m/s are {}'.format(self.id, self.MAX_UAV_VELOCITY, self.uav_radial_velocity_levels))

        # Nothing to return...

    def __f_z(self, z):
        """
        Calculate the value of $f(Z)$, i.e., the Shannon-Hartley Channel Capacity in terms of the variable $Z$

        :param z: The optimization variable in the re-formulated primal rate optimization problem

        :return: The value of the function $f(Z)$ in the re-formulated primal rate optimization problem
        """
        return self.CHANNEL_BANDWIDTH * math.log2(1 + (0.5 * math.pow(z, 2)))

    @staticmethod
    # Utility Routine
    def marcum_q(df, nc, x):
        """
        Non Odin-specific notation:

        Calculate the value of the Marcum-Q function using the specified values of the number of degrees of freedom $k$,
        the non-centrality factory $\lambda$, and the random variable rendition $x$

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
                  function--along with its associated args (df, nc, x)
        :param args: The number of degrees of freedom (df), the non-centrality parameter (nc), and an incomplete random
                     variable rendition (y) of the non-central chi-squared distribution used to evaluate the
                     Marcum-Q function

        :return: The value of the rate-adaptation primal objective function after plugging in the provided value of the
                 optimization variable ($Z$) in a certain stage of the bisection method--along with its associated
                 args (df, nc, y)
        """
        df, nc, y = args
        f_z, q_m = self.__f_z(z), self.marcum_q(df, nc, (y * (z ** 2)))
        ln_f_z = math.log(f_z) if f_z != 0.0 else np.inf
        ln_q_m = math.log(q_m) if q_m != 0.0 else np.inf
        return (-1 * ln_f_z) - ln_q_m

    @staticmethod
    # Utility Routine
    def bisect(f, df, nc, y, low, high, tolerance):
        """
        Non-Odin specific notation:

        A utility method to perform the bisection method

        :param f: The objective function being optimized
        :param df: The number of degrees of freedom ($k$) for the non-central chi-squared distribution
        :param nc: The non-centrality parameter ($\lambda$) for the non-central chi-squared distribution
        :param y: An incomplete version random variable rendition ($y$) for CDF evaluation of the non-central
                  chi-squared distribution, i.e., $x{=}y Z^{2}$
        :param low: The lower bound of the function's domain
        :param high: The upper bound of the function's domain
        :param tolerance: The tolerance level--which when achieved should terminate the optimization

        :return: The minimizer of the provided objective function, i.e., argmin

        :raises AssertionError: Assertion failed.
        """
        args = (df, nc, y)

        # assert (f(low, *args) * f(high, *args)) <= 0.0

        assert tolerance is not None
        mid = 0.0

        while abs(high - low) >= tolerance:
            mid = (high + low) / 2
            if (f(low, *args) * f(high, *args)) > 0.0:
                low = mid
            else:
                high = mid

        return mid

    def __z(self, gamma):
        """
        Calculate the value of the optimization variable $Z$

        :param gamma: The data rate ($\mathbf{\gamma}$) of Tx--employed in the calculation of $Z$

        :return: The value of the optimization variable
                 $Z{=}\sqrt{\frac{2 \beta P}{\sigma^{2} \Gamma} u(\mathbf{\gamma}, \beta)}$
        """
        return math.sqrt(2 * (math.pow((gamma / self.CHANNEL_BANDWIDTH), 2) - 1.0))

    def __u(self, gamma, d, los):
        """
        Calculate the value of the function $u(\mathbf{\gamma}, \beta)$

        :param gamma: The data rate ($\mathbf{\gamma}$) of Tx--employed to find $Z$ through $u(\mathbf{\gamma}, \beta)$
        :param d: The distance between the Tx and Rx nodes for determining the large-scale channel variations term
                  $\beta$ for use in the calculation of this function $u(\mathbf{\gamma}, \beta)$
        :param los: This boolean member will be True if the calculation of this function is for LoS settings

        :return: The value of the function $u(\mathbf{\gamma}, \beta)$
        """
        return (math.pow(2, (gamma / self.CHANNEL_BANDWIDTH)) - 1) / (self.REFERENCE_SNR_AT_1_METER *
                                                                      (d ** (-1 *
                                                                             (lambda: self.NLoS_PATH_LOSS_EXPONENT,
                                                                              lambda: self.LoS_PATH_LOSS_EXPONENT
                                                                              )[los]())))

    def __evaluate_los_throughput(self, p, d, phi, r_bar):
        """
        Calculate the contribution of LoS transmissions to the total link throughput

        :param p: The probability of LoS transmissions, i.e., $\mathbb{P}_{\text{LoS}}(\phi)$
        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                    (GU, UB, GB)
        :param r_bar: The link throughput fed into this routine to capture the contribution of LoS transmissions to
                      the total link throughput
        """
        k = self.LoS_RICIAN_FACTOR_1 * math.exp(self.LoS_RICIAN_FACTOR_2 * phi)
        df, nc = 2, (2 * k)
        y = (k + 1) * (1 / (self.REFERENCE_SNR_AT_1_METER * (d ** (-1 * self.LoS_PATH_LOSS_EXPONENT))))
        z_star = self.bisect(self.__f, df, nc, y, 0, self.__z((self.CHANNEL_BANDWIDTH *
                                                               math.log2(1 + (math.pow(rice.ppf(0.9999, k), 2) *
                                                                              self.REFERENCE_SNR_AT_1_METER *
                                                                              (d ** (-1 * self.LoS_PATH_LOSS_EXPONENT)))
                                                                         )) + 100.0), self.BISECTION_METHOD_TOLERANCE)
        gamma_star = self.__f_z(z_star)
        x_star = 2 * (k + 1) * self.__u(gamma_star, d, True)
        r_star = gamma_star * self.marcum_q(df, nc, x_star)

        # with self.lock:
        #     r_bar += p * r_star

        try:
            tf.compat.v1.assign(r_bar, r_bar + (p * r_star), validate_shape=True, use_locking=True)
        except Exception as e:
            print('[ERROR] [{}] SMDPEvaluation __evaluate_los_throughput: Exception caught during tensor assignment - '
                  '{}'.format(self.id, traceback.print_tb(e.__traceback__)))

        # Nothing to return...

    def __evaluate_nlos_throughput(self, p, d, r_bar):
        """
        Calculate the contribution of NLoS transmissions to the total link throughput

        :param p: The probability of NLoS transmissions, i.e., $\mathbb{P}_{\text{NLoS}}(\phi)$
        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param r_bar: The link throughput fed into this routine to capture the contribution of NLoS transmissions to
                      the total link throughput
        """
        df, nc = 2, 0
        y = 1 / (self.REFERENCE_SNR_AT_1_METER * (self.NLoS_ATTENUATION_CONSTANT *
                                                  d ** (-1 * self.NLoS_PATH_LOSS_EXPONENT)))
        z_star = self.bisect(self.__f, df, nc, y, 0, self.__z((self.CHANNEL_BANDWIDTH *
                                                               math.log2(1 + (math.pow(rayleigh.ppf(0.9999), 2) *
                                                                              self.REFERENCE_SNR_AT_1_METER *
                                                                              (self.NLoS_ATTENUATION_CONSTANT *
                                                                               d ** (-1 * self.NLoS_PATH_LOSS_EXPONENT))
                                                                              ))) + 100.0),
                             self.BISECTION_METHOD_TOLERANCE)
        gamma_star = self.__f_z(z_star)
        x_star = 2 * self.__u(gamma_star, d, False)
        r_star = gamma_star * self.marcum_q(df, nc, x_star)

        # with self.lock:
        #     r_bar += p * r_star

        # Try-Catch Block to handle resource access exceptions
        try:
            tf.compat.v1.assign(r_bar, r_bar + (p * r_star), validate_shape=True, use_locking=True)
        except Exception as e:
            print('[ERROR] [{}] SMDPEvaluation __evaluate_los_throughput: Exception caught during tensor assignment - '
                  '{}'.format(self.id, traceback.print_tb(e.__traceback__)))

        # Nothing to return...

    def __calculate_adapted_throughput(self, d, phi, r_bar):
        """
        Calculate the throughput of the link under analysis (GU, UB, GB) post rate-adaptation in $Z$

        :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
        :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                    (GU, UB, GB)
        :param r_bar: The link throughput fed into this routine for reference-based concurrent updates

        :return: The rate-adapted throughput for the uplink link under analysis (GU, UB, GB)
        """
        p = 1 / (1 + (self.PROPAGATION_ENVIRONMENT_PARAMETER_1 *
                      math.exp(-1 * self.PROPAGATION_ENVIRONMENT_PARAMETER_2 *
                               (phi - self.PROPAGATION_ENVIRONMENT_PARAMETER_1))))

        with ThreadPoolExecutor(max_workers=40000) as executor:
            executor.submit(self.__evaluate_los_throughput, p, d, phi, r_bar)
            executor.submit(self.__evaluate_nlos_throughput, (1 - p), d, r_bar)

        # Nothing to return...

    @staticmethod
    # Utility Routine
    def interpolate_waypoints(p_indices, p, res_multiplier):
        """
        Increase the resolution of the provided trajectory solution, i.e., increase the number of waypoints
        (called from both HCSO and CSO)

        :param p_indices: The array of indices for referencing the trajectory solution during scipy splining
        :param p: The trajectory solution whose resolution is to be improved
        :param res_multiplier: The number of waypoints in the returned zip object would be res_multiple times the number
                               of waypoints in the original

        :return: A zip object containing the interpolated x and y coordinates of the trajectory solution
        """
        m = len(p_indices)
        spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)
        return tf.constant(list(zip(spl_x(np.linspace(0, (m - 1), (res_multiplier * m), dtype=np.float64)),
                                    spl_y(np.linspace(0, (m - 1), (res_multiplier * m), dtype=np.float64)))))

    @staticmethod
    # Utility Routine
    def interpolate_velocities(v_indices, v, res_multiplier):
        """
        Increase the resolution of the provided trajectory solution, i.e., increase the number of waypoints
        (called from both HCSO and CSO)

        :param v_indices: The array of indices for referencing the UAV velocities solution during scipy splining
        :param v: The UAV velocities solution whose resolution is to be improved
        :param res_multiplier: The number of UAV velocities in the returned zip object would be res_multiple times the
                               number of UAV velocities in the original

        :return: A zip object containing the interpolated velocities from the original UAV velocities solution
        """
        m, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
        return spl_v(np.linspace(0, (m - 1), (res_multiplier * m), dtype=np.float64))

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
        p = self.interpolate_waypoints([_ for _ in range(p__.shape[0])], p__, res_multiplier)
        midpoint = int(p.shape[0] / 2)
        v = self.interpolate_velocities([_ for _ in range(v__.shape[0])], v__, res_multiplier)
        t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1), v[:-1])

        # Decode (GN - UAV) | Forward (UAV - BS)
        with strategy.scope():
            r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
            h_u = tf.constant(self.UAV_HEIGHT, shape=r_gu.shape, dtype=tf.float64)
            d_gu = tf.norm(tf.concat([r_gu[:, None], h_u[:, None]], axis=1), axis=1)
            phi_gu = tf.asin(tf.divide(h_u, d_gu))
            r_bar_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

            with ThreadPoolExecutor(max_workers=(4000 * d_gu.shape[0])) as executor:
                for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
                    executor.submit(self.__calculate_adapted_throughput, d_gu__, phi_gu__, r_bar_gu[i__])

            h_1 = self.packet_length - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))
            t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
            e_p_1 = (lambda: 0.0, lambda: (self.evaluate_power_consumption(22.0) * t_p_1))[h_1.numpy() > 0.0]()

            r_ub = tf.norm(p[midpoint:], axis=1)
            h_ub = tf.constant(abs(self.UAV_HEIGHT - self.BASE_STATION_HEIGHT), shape=r_ub.shape, dtype=tf.float64)
            d_ub = tf.norm(tf.concat([r_ub[:, None], h_ub[:, None]], axis=1), axis=1)
            phi_ub = tf.asin(tf.divide(h_ub, d_ub))
            r_bar_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)

            with ThreadPoolExecutor(max_workers=(4000 * d_ub.shape[0])) as executor:
                for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
                    executor.submit(self.__calculate_adapted_throughput, d_ub__, phi_ub__, r_bar_ub[i__])

            h_2 = self.packet_length - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))
            t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
            e_p_2 = (lambda: 0.0, lambda: (self.evaluate_power_consumption(22.0) * t_p_2))[h_2.numpy() > 0.0]()

        return self.PENALTIES_CAPSULE(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)

    @tf.function
    def __power_cost(self, v, num_workers):
        """
        A tf function decorated routine to enable faster application of the power consumption evaluation method on the
        UAV velocity tensor

        :param v: The UAV flying velocity tensor against which the power consumption evaluation is to be performed
        :param num_workers: The number of parallel runs allowed/considered during this power consumption evaluation on v

        :return: [tf.map_fn] on [self.evaluate_power_consumption] with [parallel_iterations = 2 x num_workers]
        """
        return tf.map_fn(self.evaluate_power_consumption, v, parallel_iterations=(2 * num_workers))

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
        t_p_1, t_p_2, e_p_1, e_p_2 = self.__penalties(p, v, x_g, self.INTERPOLATION_FACTOR)
        t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1), v[:-1])
        e = tf.multiply(t, self.__power_cost(v[:-1], v[:-1].shape[0]))
        e__, t__ = tf.reduce_sum(t), tf.reduce_sum(e)
        f_hat += ((1.0 - (nu * self.average_power_constraint)) * (t_p_1 + t_p_2 + t__)) + \
                 (nu * (e_p_1 + e_p_2 + e__))

        # Energy and Time Duration cost updates outside HCSO--inside SMDP VITER
        if e_hat is not None:
            e_hat += e__.numpy()

        if t_hat is not None:
            t_hat += t__.numpy()
        # Nothing to return...

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
        f_hat_t_j, f_hat_t_j_1 = 0.0, 0.0
        p_t_j, p_t_j_1, v_t_j, v_t_j_1 = p[t_j], p[t_j_1], v[t_j], v[t_j_1]
        u_t_j, u_t_j_1, w_t_j, w_t_j_1 = u[t_j], u[t_j_1], w[t_j], w[t_j_1]

        with ThreadPoolExecutor(max_workers=40000) as executor:
            executor.submit(self.__calculate_comm_cost, p_t_j, v_t_j, nu, x_g, f_hat_t_j)

            executor.submit(self.__calculate_comm_cost, p_t_j_1, v_t_j_1, nu, x_g, f_hat_t_j_1)

        argmin__ = np.argmin([f_hat_t_j, f_hat_t_j_1])
        j_win, p_w, v_w, j_los, p_l, v_l, u_l, w_l = (t_j, p_t_j, v_t_j, t_j_1, p_t_j_1, v_t_j_1, u_t_j_1, w_t_j_1) \
            if (argmin__ == 0) else (t_j_1, p_t_j_1, v_t_j_1, t_j, p_t_j, v_t_j, u_t_j, w_t_j)
        r_j = np.random.uniform(size=[1, 3])

        # Device placement: Trajectory particles | UAV velocity particles
        with strategy.scope():
            tf.compat.v1.assign(f_hats[t_j], f_hat_t_j, validate_shape=True, use_locking=False)
            u_j_los = tf.add((r_j[0] * u_l), (r_j[1] * tf.subtract(p_w, p_l)),
                             (self.CSO_PARTICLE_VELOCITY_SCALING_FACTOR * r_j[2] * tf.subtract(p_bar, p_l)))
            p_j_los = tf.add(p_l, u_j_los)
            tf.compat.v1.assign(p[j_los], p_j_los, validate_shape=True, use_locking=False)
            tf.compat.v1.assign(u[j_los], u_j_los, validate_shape=True, use_locking=False)

            tf.compat.v1.assign(f_hats[t_j_1], f_hat_t_j_1, validate_shape=True, use_locking=False)
            w_j_los = tf.add((r_j[0] * w_l), (r_j[1] * tf.subtract(v_w, v_l)),
                             (self.CSO_PARTICLE_VELOCITY_SCALING_FACTOR * r_j[2] * tf.subtract(v_bar, v_l)))
            v_j_los = tf.clip_by_value(tf.add(v_l, w_j_los), self.CSO_MINIMUM_VELOCITY_VALUE, self.MAX_UAV_VELOCITY)
            tf.compat.v1.assign(v[j_los], v_j_los, validate_shape=True, use_locking=False)
            tf.compat.v1.assign(w[j_los], w_j_los, validate_shape=True, use_locking=False)

        # Nothing to return...

    # Core Descriptive Routine
    def __competitive_swarm_optimization(self, initial_uav_position, terminal_uav_position, ground_node_position,
                                         trajectory_particles, uav_velocity_particles,
                                         trajectory_particle_velocities, uav_velocity_particle_velocities,
                                         swarm_size, segment_size, dual_variable):
        """
        The Competitive Swarm Optimization (CSO) Algorithm for UAV trajectory optimization in the communication states

        :param initial_uav_position: The initial UAV position in the form $\mathbf{x}_{0}{=}(r_{U}, 0)$
        :param terminal_uav_position: The final UAV position in the form
                                      $\mathbf{x}_{M}{=}\hat{r}_{U}\frac{\mathbf{x}_{M-1}}{||\mathbf{x}_{M-1}||_{2}}$,
                                      if $||\mathbf{x}_{M-1}||_{2}{\neq}0$, else $\mathbf{x}_{M}{=}(\hat{r}_{U}, 0)$
        :param ground_node_position: The position of the Ground Node (GN) generating the communication request
        :param trajectory_particles: The NxM matrix of possible trajectory solutions for iterative HCSO-CSO
                                     optimization ($\mathbf{p}_{[1:N]}$) [An NxM tensorflow matrix]
        :param uav_velocity_particles: The NxM matrix of possible UAV velocity solutions for iterative HCSO-CSO
                                       optimization ($\mathbf{v}_{[1:N]}$) [An NxM tensorflow matrix]
        :param trajectory_particle_velocities: The particle velocities of the provided trajectory solutions
                                               ($\mathbf{u}_{[1:N]}$) [An NxM tensorflow matrix]
        :param uav_velocity_particle_velocities: The particle velocities of the provided UAV velocity solutions
                                                 ($\mathbf{w}_{[1:N]}$) [An NxM tensorflow matrix]
        :param swarm_size: The number of trajectory & UAV velocity particles passed down from HCSO ($N$)
        :param segment_size: The size of each trajectory & UAV velocity particle passed down from HCSO ($M$)
        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization

        :return: The winning trajectory & UAV velocity particles being returned to HCSO--along with the associated cost
        """
        x_0, x_m, x_g = initial_uav_position, terminal_uav_position, ground_node_position
        p, v = trajectory_particles, uav_velocity_particles
        p_bar, v_bar = tf.reduce_mean(p, axis=0), tf.reduce_mean(v, axis=0)
        u, w = trajectory_particle_velocities, uav_velocity_particle_velocities
        k, n, m, nu = 0, swarm_size, segment_size, dual_variable
        f_hats = tf.Variable(tf.zeros(shape=[n, ], dtype=tf.float64), dtype=tf.float64)
        indices = [_ for _ in range(n)]

        while k < self.MAXIMUM_COST_EVALUATIONS:
            t = tf.random.shuffle(indices)
            with ThreadPoolExecutor(max_workers=(2000 * n)) as executor:
                for j in range(0, n, 2):
                    executor.submit(self.__update_winners_and_losers, p, v, u, w, t[j], t[j + 1], p_bar, v_bar, f_hats,
                                    nu, x_g)
            k += 1

        i = min(indices, key=lambda i__: f_hats[i__].numpy())
        return p[i], v[i], f_hats[i]

    # Core Descriptive Routine
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
        id__, p_avg, delta_0, = self.id, self.average_power_constraint, self.WAITING_STATE_INTERVAL
        i__, j__, r_u, v_r, nu = state_index, action_index, uav_position.numpy(), radial_velocity.numpy(), dual_variable
        l_wait_star, e_wait_star, t_wait_star = lagrangians, energies, time_durations
        alpha_c, delta_c = self.ANGULAR_VELOCITY_LEARNING_RATE, self.ANGULAR_VELOCITY_TERMINATION_THRESHOLD
        conf, conf_th, theta_c_1, theta_c = 0, self.CONVERGENCE_CONFIDENCE, tf.Variable(np.inf), tf.Variable(0.0)

        # The optimization problem
        with UAVAngularVelocityOptimization(id__, self, nu, r_u, v_r, theta_c) as problem:
            # The optimizer
            optimizer = tfco.LagrangianOptimizer(optimizer=tf.optimizers.Adagrad(learning_rate=alpha_c),
                                                 num_constraints=problem.num_constraints)
            # The list of variables relevant to the optimization
            var_list = [theta_c, problem.trainable_variables, optimizer.trainable_variables()]
            # Optimize
            while (abs(theta_c - theta_c_1) >= delta_c) or (conf < conf_th):
                theta_c_1 = theta_c
                optimizer.minimize(problem, var_list=var_list)
                conf += 1

        theta_c_star = theta_c.numpy()
        p_mob_star = self.evaluate_power_consumption(((v_r ** 2) + ((r_u ** 2) + (theta_c_star ** 2))) ** 0.5)
        tf.compat.v1.assign(l_wait_star[i__, j__], (nu * (p_mob_star - p_avg) * delta_0), validate_shape=True,
                            use_locking=False)
        tf.compat.v1.assign(e_wait_star[i__, j__], (p_mob_star * delta_0), validate_shape=True, use_locking=False)
        tf.compat.v1.assign(t_wait_star[i__, j__], delta_0, validate_shape=True, use_locking=False)

        # Nothing to return...

    # Core Descriptive Routine
    def __hierarchical_competitive_swarm_optimization(self, state_index, action_index, initial_uav_position,
                                                      terminal_uav_position, dual_variable, ground_node_position,
                                                      lagrangians, energies, time_durations):
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
        :param lagrangians: The Lagrangian cost metrics collection for a value update associated with the fed-in comm
                            state and action [concurrency requirement]
        :param energies: The energy cost metrics collection for a value update associated with the fed-in comm
                         state and action [concurrency requirement]
        :param time_durations: The time-duration cost metrics collection for a value update associated with the fed-in
                               comm state and action [concurrency requirement]
        """
        eval__ = True if (state_index == self.canary_state_index) else False  # Do I need to log Lagrangian costs?
        i__, j__, xi_s, opt_traj = state_index, action_index, self.relay_status, self.optimal_trajectories
        l_comm_star, e_comm_star, t_comm_star = lagrangians, energies, time_durations
        id__, i, nu, f_hat, e_usage, delta, r_bar = self.id, 0, dual_variable, 0.0, 0.0, 0.0, 0.0
        x_0, x_m, x_g, h_b = initial_uav_position, terminal_uav_position, ground_node_position, self.BASE_STATION_HEIGHT
        a, v_levels, p_len = self.CELL_RADIUS, self.CSO_VELOCITY_DISCRETIZATION_LEVELS, self.packet_length
        n, m_old, m_ip = self.INITIAL_NUMBER_OF_PARTICLES, self.INITIAL_TRAJECTORY_SEGMENTS, self.INTERPOLATION_FACTOR
        v_min, v_max, eps = self.CSO_MINIMUM_VELOCITY_VALUE, self.MAX_UAV_VELOCITY, self.HCSO_VELOCITY_SCALING_FACTOR
        zeta, m = self.HCSO_TRAJECTORY_SCALING_FACTOR, (m_ip * (m_old + 2))

        # Device Placement: Random trajectories' generation & subsequent interpolation for HCSO initialization
        with strategy.scope():
            with RandomTrajectoriesGeneration(id__, x_0, x_m, ((-1 * a), a), (0, (2 * np.pi)),
                                              n, m_old, m_ip) as generator:
                p = generator.optimize(generator.generate())

            # Random tensor generation for UAV velocity particles and particle velocities collections for the
            #   trajectories tensor and the UAV velocities tensor, for HCSO initialization
            v, u, w = [tf.Variable(np.random.choice(np.linspace(v_min, v_max, v_levels, dtype=np.float64),
                                                    size=[n, m]))] * 3

        # HCSO while loop
        while m <= self.MAXIMUM_TRAJECTORY_SEGMENTS:
            p_star, v_star, f_star = self.__competitive_swarm_optimization(x_0, x_m, x_g, p, v, u, w, n, m, nu)  # CSO

            # Save the Lagrangian cost metric's value at this stage for Cost-Computation Convergence
            if eval__:
                self.canary_state_comm_costs[time.monotonic()] = f_star

            n -= (2 * (i + 1))  # Update swarm size
            indices = [_ for _ in range(m)]

            # Device Placement: Trajectory particles | UAV velocity particles
            with strategy.scope():
                p_tilde = tf.tile(tf.expand_dims(self.interpolate_waypoints(indices, p_star, m_ip), axis=0),
                                  multiples=[n, 1, 1])
                p = tf.Variable(tf.zeros(shape=p_tilde.shape, dtype=tf.float64), dtype=tf.float64)
                p_1, p_2 = tf.roll(p_tilde, shift=-1, axis=1)[:, :-1, :], p_tilde[:, :-1, :]
                scale = tf.expand_dims(tf.add(tf.square(tf.norm(tf.subtract(p_1, p_2), axis=2)),
                                              tf.multiply(zeta, tf.square(tf.norm(tf.subtract(p_2, p_1), axis=2)))),
                                       axis=2)
                scales, shape_p = tf.tile(scale, multiples=[1, 1, 2]), [n, (m_ip * m) - 1, 2]
                scale_last = tf.expand_dims(tf.add(tf.square(tf.norm(p_tilde[:, -1, :], axis=1)),
                                                   tf.multiply(zeta, tf.square(tf.norm(tf.subtract(p_tilde[:, -2, :],
                                                                                                   p_tilde[:, -1, :]),
                                                                                       axis=1)))), axis=1)

                p_tilde_r = tf.random.normal(shape=shape_p, mean=tf.zeros(shape=shape_p, dtype=tf.float64),
                                             stddev=tf.sqrt(scales), dtype=tf.float64)
                tf.compat.v1.assign(p[:, :-1, :], tf.add(p_tilde[:, :-1, :], p_tilde_r),
                                    validate_shape=True, use_locking=False)
                tf.compat.v1.assign(p[:, -1, :], tf.add(p_tilde[:, -1, :],
                                                        tf.random.normal(shape=[scale_last.shape[0], 2],
                                                                         mean=tf.zeros(shape=[scale_last.shape[0], 2],
                                                                                       dtype=tf.float64),
                                                                         stddev=tf.tile(tf.sqrt(scale_last),
                                                                                        multiples=[1, 2]),
                                                                         dtype=tf.float64)),
                                    validate_shape=True, use_locking=False)

                shape_v = [n, (m_ip * m)]
                v_tilde = tf.tile(tf.expand_dims(self.interpolate_velocities(indices, v_star, m_ip), axis=0),
                                  multiples=[n, 1])
                v_tilde_r = tf.random.normal(shape=shape_v, mean=tf.zeros(shape=shape_v, dtype=tf.float64),
                                             stddev=tf.sqrt(tf.multiply(eps * ((v_max - v_min) ** 2),
                                                                        tf.ones(shape=shape_v, dtype=tf.float64))),
                                             dtype=tf.float64)
                v = tf.clip_by_value(tf.add(v_tilde, v_tilde_r), v_min, v_max)

            m *= m_ip  # Update segment size

        p_star_2, v_star_2, f_star_2 = self.__competitive_swarm_optimization(x_0, x_m, x_g, p, v, u, w, n, m, nu)  # CSO
        # Save the Lagrangian cost metric's value at this stage for Cost-Computation Convergence
        if eval__:
            self.canary_state_comm_costs[time.monotonic()] = f_star_2

        d_gb = tf.norm(tf.concat([h_b, tf.norm(x_g, axis=0)], axis=0), axis=0)
        phi_gb = tf.asin(tf.divide(h_b, d_gb))

        # Lagrangian cost determination for scheduling (Direct BS or UAV relay?)
        with ThreadPoolExecutor(max_workers=40000) as executor:
            executor.submit(self.__calculate_comm_cost, p_star_2, v_star_2, nu, x_g, f_hat, e_usage, delta)

            executor.submit(self.__calculate_adapted_throughput, d_gb, phi_gb, tf.Variable(r_bar, dtype=tf.float64))

        l_xi_0, e_xi_0, t_xi_0 = (p_len / r_bar) if r_bar != 0.0 else np.inf, 0.0, 0.0
        l_xi_1, e_xi_1, t_xi_1 = f_hat, e_usage, delta
        l__, e__, t__, xi__ = (l_xi_1, e_xi_1, t_xi_1, 1) if (l_xi_1 < l_xi_0) else (l_xi_0, 0.0, 0.0, 0)

        # Device Placement: Tensor updates for further processing | Class-wide data collection updates for visualization
        with strategy.scope():
            tf.compat.v1.assign(l_comm_star[i__, j__], l__, validate_shape=True, use_locking=False)
            tf.compat.v1.assign(e_comm_star[i__, j__], e__, validate_shape=True, use_locking=False)
            tf.compat.v1.assign(t_comm_star[i__, j__], t__, validate_shape=True, use_locking=False)

            tf.compat.v1.assign(xi_s[i__], xi__, validate_shape=True, use_locking=False)
            tf.compat.v1.assign(opt_traj[i__, j__], p_star_2, validate_shape=True, use_locking=False)

        # Nothing to return...

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
        with ThreadPoolExecutor(max_workers=(2000 * s_wait.shape[0] * a_wait.shape[0])) as executor:
            for i__, r_u in enumerate(s_wait):
                for j__, v_r in enumerate(a_wait):
                    executor.submit(self.__angular_velocity_optimization, i__, j__, r_u, v_r, nu, l_wait_star,
                                    e_wait_star, t_wait_star)

        # Nothing to return...

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
        with ThreadPoolExecutor(max_workers=(4000 * s_comm.shape[0] * a_comm.shape[0])) as executor:
            for i__, s in enumerate(s_comm):
                x_0, x_g = s
                for j__, x_u in enumerate(a_comm):
                    executor.submit(self.__hierarchical_competitive_swarm_optimization, i__, j__, x_0, x_u,
                                    nu, x_g, l_comm_star, e_comm_star, t_comm_star)

        # Nothing to return...

    def __smdp_waiting_viter_updates(self, l_wait_star, e_wait_star, t_wait_star, v_i_wait, v_i_comm, h_i_wait,
                                     e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait, o_star):
        """
        The updates to the value functions of waiting states are performed in this routine--along with similar updates
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
        lambda__, delta_0 = self.ARRIVAL_RATE, self.WAITING_STATE_INTERVAL
        v_i_1_wait = tf.constant(v_i_wait)

        s_wait_added = tf.add(tf.tile(tf.expand_dims(s_wait, axis=1), multiples=[1, a_wait.shape[0]]),
                              (delta_0 * a_wait))
        wait_indices = tf.argmin(tf.abs(tf.subtract(tf.tile(tf.expand_dims(s_wait_added, axis=2),
                                                            multiples=[1, 1, s_wait.shape[0]]), s_wait)), axis=2)

        # Device placement: Waiting state value functions | Comm state value functions
        with strategy.scope():
            v_i_wait_remapped = tf.gather(tf.tile(tf.expand_dims(v_i_wait, axis=1), multiples=[1, a_wait.shape[0]]),
                                          wait_indices)[:, :, 0]

            mins_org = tf.reshape(tf.gather(s_wait, wait_indices), shape=[-1])
            v_i_comm_remapped = tf.reshape(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(v_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org), axis=1),
                shape=[s_wait.shape[0], a_wait.shape[0]])

        _v_i_added = tf.add(l_wait_star, (math.exp(-1 * lambda__ * delta_0) * v_i_wait_remapped),
                            ((1 - math.exp(-1 * lambda__ * delta_0)) * v_i_comm_remapped))
        o_i_1 = tf.argmin(_v_i_added, axis=1)
        o_star__ = tf.gather(a_wait, o_i_1)

        tf.compat.v1.assign(v_i_wait, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
        tf.compat.v1.assign(h_i_wait, tf.subtract(v_i_wait, v_i_1_wait), validate_shape=True, use_locking=False)
        tf.compat.v1.assign(o_star, o_star__, validate_shape=True, use_locking=False)

        s_wait_added_min = tf.add(s_wait, (delta_0 * o_star__))
        wait_indices_min = tf.squeeze(tf.argmin(tf.abs(tf.subtract(tf.tile(tf.expand_dims(s_wait_added_min, axis=1),
                                                                           multiples=[1, s_wait.shape[0]]), s_wait)),
                                                axis=1))

        # Device placement: Waiting state energy & time costs | Comm state energy & time costs
        with strategy.scope():
            e_wait_star_remapped = tf.boolean_mask(e_wait_star, tf.one_hot(o_i_1, e_wait_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            e_i_wait_remapped = tf.gather(e_i_wait, wait_indices_min)
            t_wait_star_remapped = tf.boolean_mask(t_wait_star, tf.one_hot(o_i_1, t_wait_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            t_i_wait_remapped = tf.gather(t_i_wait, wait_indices_min)

            mins_org_min = tf.gather(s_wait, wait_indices_min)
            e_i_comm_remapped = tf.squeeze(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(e_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org_min),
                axis=1))
            t_i_comm_remapped = tf.squeeze(tf.reduce_sum(tf.map_fn(
                lambda x: tf.gather(t_i_comm, tf.where(tf.equal(tf.norm(s_comm[:, 0], axis=1), x))), mins_org_min),
                axis=1))

        # Device placement: Energy cost | Delay (time duration) cost
        with strategy.scope():
            _e_i_added = tf.add(e_wait_star_remapped, (math.exp(-1 * lambda__ * delta_0) * e_i_wait_remapped),
                                ((1 - math.exp(-1 * lambda__ * delta_0)) * e_i_comm_remapped))
            tf.compat.v1.assign(e_i_wait, _e_i_added, validate_shape=True, use_locking=True)

            _t_i_added = tf.add(t_wait_star_remapped, (math.exp(-1 * lambda__ * delta_0) * t_i_wait_remapped),
                                ((1 - math.exp(-1 * lambda__ * delta_0)) * t_i_comm_remapped))
            tf.compat.v1.assign(t_i_wait, _t_i_added, validate_shape=True, use_locking=True)

        # Nothing to return

    # noinspection PyMethodMayBeStatic
    def __smdp_comm_viter_updates(self, l_comm_star, e_comm_star, t_comm_star, v_i_comm, v_i_wait, h_i_comm,
                                  e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm, u_star):
        """
        The updates to the value functions of comm states are performed in this routine--along with similar updates
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
        """
        v_i_1_comm = tf.constant(v_i_comm)

        _v_i_wait = tf.tile(tf.expand_dims(tf.squeeze(tf.map_fn(lambda x: tf.gather(v_i_wait,
                                                                                    tf.where(tf.equal(s_wait, x))),
                                                                tf.norm(a_comm, axis=1))), axis=1),
                            multiples=[1, l_comm_star.shape[0]])
        _v_i_added = tf.add(l_comm_star, _v_i_wait)  # Transition only to the waiting state

        u_i_1 = tf.math.argmin(_v_i_added, axis=1)
        u_star__ = tf.gather(a_comm, u_i_1, axis=0)

        tf.compat.v1.assign(v_i_comm, tf.reduce_min(_v_i_added, axis=1), validate_shape=True, use_locking=True)
        tf.compat.v1.assign(h_i_comm, tf.subtract(v_i_comm, v_i_1_comm), validate_shape=True, use_locking=False)
        tf.compat.v1.assign(u_star, u_star__, validate_shape=True, use_locking=False)

        # Device Placement: Comm state energy costs | Comm state time duration costs
        with strategy.scope():
            e_i_wait_remapped = tf.squeeze(tf.map_fn(lambda x: tf.gather(e_i_wait, tf.where(tf.equal(s_wait, x))),
                                                     tf.norm(u_star__, axis=1)))
            e_comm_star_remapped = tf.boolean_mask(e_comm_star, tf.one_hot(u_i_1, e_comm_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            tf.compat.v1.assign(e_i_comm, tf.add(e_comm_star_remapped, e_i_wait_remapped), validate_shape=True,
                                use_locking=True)

            t_i_wait_remapped = tf.squeeze(tf.map_fn(lambda x: tf.gather(t_i_wait, tf.where(tf.equal(s_wait, x))),
                                                     tf.norm(u_star__, axis=1)))
            t_comm_star_remapped = tf.boolean_mask(t_comm_star, tf.one_hot(u_i_1, t_comm_star.shape[1], on_value=True,
                                                                           off_value=False, dtype=tf.bool))
            tf.compat.v1.assign(t_i_comm, tf.add(t_comm_star_remapped, t_i_wait_remapped), validate_shape=True,
                                use_locking=True)

        # Nothing to return

    def __pi_comm(self):
        """
        A simple routine to evaluate the steady state probability of being in the communication state
        ($\pi_{\text{comm}}{=}\frac{1 - e^{-\Lambda \Delta_{0}}}{2 - e^{-\Lambda \Delta_{0}}})

        :return: The steady state probability of being in the communication state ($\pi_{\text{comm}}$)
        """
        p_e = math.exp(-1 * self.ARRIVAL_RATE * self.WAITING_STATE_INTERVAL)
        return (1 - p_e) / (2 - p_e)

    # Core Descriptive Routine
    def __value_iteration(self, dual_variable):
        """
        The SMDP Value Iteration Algorithm for optimal waiting state & communication state policy determination, given
        a particular value of the dual variable for Lagrangian cost metric evaluation

        :param dual_variable: The value of the dual variable ($\nu$) in a particular iteration of dual maximization

        :return: The SMDP_VITER_DATA_CAPSULE namedtuple constituting the following members:
                 $(O_{k}^{*}, U_{k}^{*}, g_{k}, \Bar{E}_{k}, \Bar{T}_{k})$
        """
        i, nu, delta, canary_index = 0, dual_variable, self.VITER_TERMINATION_THRESHOLD, self.canary_state_index
        conf, conf_th = 0, self.CONVERGENCE_CONFIDENCE

        # Device Placement: Waiting states | Comm states
        with strategy.scope():
            s_wait, a_wait = self.waiting_states, self.waiting_actions
            s_wait_size, a_wait_size = s_wait.shape[0], a_wait.shape[0]
            l_wait_star, e_wait_star, t_wait_star = [tf.Variable(tf.zeros(shape=[s_wait_size, a_wait_size],
                                                                          dtype=tf.float64), dtype=tf.float64)] * 3
            v_i_wait, h_i_wait, e_i_wait, t_i_wait = [tf.Variable(tf.zeros(shape=[s_wait_size, ],
                                                                           dtype=tf.float64), dtype=tf.float64)] * 4
            o_star = tf.Variable(tf.zeros(shape=[s_wait_size, ], dtype=tf.float64), dtype=tf.float64)

            s_comm, a_comm = self.comm_states, self.comm_actions
            s_comm_size, a_comm_size = s_comm.shape[0], a_comm.shape[0]
            l_comm_star, e_comm_star, t_comm_star = [tf.Variable(tf.zeros(shape=[s_comm_size, a_comm_size],
                                                                          dtype=tf.float64), dtype=tf.float64)] * 3
            v_i_comm, h_i_comm, e_i_comm, t_i_comm = [tf.Variable(tf.zeros(shape=[s_comm_size, ],
                                                                           dtype=tf.float64), dtype=tf.float64)] * 4
            u_star = tf.Variable(tf.zeros(shape=[s_comm_size, 2], dtype=tf.float64), dtype=tf.float64)

        max_workers_o = 8000 * ((s_wait_size * a_wait_size) + (s_comm_size * a_comm_size))
        max_workers_a = 2000 * (s_wait_size + s_comm_size)

        # Optimization (lagrangian cost [via optimal action] determination per state (both waiting & comm states)
        with ThreadPoolExecutor(max_workers=max_workers_o) as executor:
            executor.submit(self.__optimize_waiting_states, nu, s_wait, a_wait, l_wait_star, e_wait_star, t_wait_star)

            executor.submit(self.__optimize_comm_states, nu, s_comm, a_comm, l_comm_star, e_comm_star, t_comm_star)

        # Value function updates (via SMDP state transitions: one step future look-ahead) (both waiting & comm states)
        while ((tf.reduce_max(h_i_wait) - tf.reduce_min(h_i_wait)) >= delta) or \
                ((tf.reduce_max(h_i_comm) - tf.reduce_min(h_i_comm)) >= delta) or conf <= conf_th:
            with ThreadPoolExecutor(max_workers=max_workers_a) as executor:
                executor.submit(self.__smdp_waiting_viter_updates, l_wait_star, e_wait_star, t_wait_star, v_i_wait,
                                v_i_comm, h_i_wait, e_i_wait, e_i_comm, t_i_wait, t_i_comm, s_wait, s_comm, a_wait,
                                o_star)

                executor.submit(self.__smdp_comm_viter_updates, l_comm_star, e_comm_star, t_comm_star, v_i_comm,
                                v_i_wait, h_i_comm, e_i_comm, e_i_wait, t_i_comm, t_i_wait, s_wait, a_comm, u_star)

            i += 1

        pi_comm = self.__pi_comm()
        g, e_bar, t_bar = ((1 / (pi_comm * i)) if pi_comm != 0.0 else np.inf) * np.array(
            [v_i_comm[canary_index].numpy(), e_comm_star[canary_index].numpy(), t_comm_star[canary_index].numpy()])
        return self.SMDPValueIterationDataCapsule(nu=nu, o_star=o_star, u_star=u_star, g=g, e_bar=e_bar, t_bar=t_bar)

    # Core Descriptive Routine
    def projected_subgradient_ascent(self):
        """
        The Projected Sub-gradient Ascent Algorithm for dual variable maximization

        Visualize: The optimal waiting state and communication state policies ($(O^{*}, U^{*})$) via the plots described
        at the top of this script [script docstring]
        """
        p_av, rho_0 = self.average_power_constraint, self.INITIAL_DUAL_VARIABLE_STEP_SIZE
        th_d, th_pf = self.DUAL_CONVERGENCE_THRESHOLD, self.PRIMAL_FEASIBILITY_THRESHOLD
        th_cs = self.COMPLEMENTARY_SLACKNESS_THRESHOLD
        k, nu_k, g_k, g_k_1, o_star_k, u_star_k, e_k, t_k = astuple(self.__value_iteration(0.0))

        while (abs(g_k - g_k_1) >= th_d) or (e_k - (p_av * t_k) >= th_pf) or (nu_k * abs(e_k - (p_av * t_k)) >= th_cs):
            g_k_1, nu_k = g_k, max(nu_k + ((rho_0 / (k + 1)) * (e_k - (p_av * t_k))), 0)
            nu_k, o_star_k, u_star_k, g_k, e_k, t_k = astuple(self.__value_iteration(nu_k))
            k += 1

        # Device Placement: Post-convergence visualizations for the optimal waiting state policy [and comm state costs]
        # with tf.device('/CPU:0'):
        #     with ThreadPoolExecutor(max_workers=40000) as executor:
        #         executor.submit(self.prepare_lagrangian_convergence_data)
        #         executor.submit(self.prepare_waiting_state_policy_data, o_star_k)
        #
        # Device Placement: Post-convergence visualizations for the optimal comm state policy [and delay]
        # with tf.device('/CPU:0'):
        #     with ThreadPoolExecutor(max_workers=60000) as executor:
        #         executor.submit(self.prepare_delay_power_tradeoff_data, t_k)
        #         executor.submit(self.prepare_no_relay_gns_data, u_star_k)
        #         executor.submit(self.prepare_uav_trajectories_data, u_star_k)

        self.o_star, self.u_star, self.canary_comm_average_delay = o_star_k, u_star_k, t_k

        # Nothing to return...

    # Core Visualization Routine
    def prepare_waiting_state_policy_data(self, waiting_policies):
        """
        UAV Radial Velocity ($v_{r}$ m/s) v UAV Radius Level ($r_{U}$ m) w.r.t the optimal waiting state policy, for
        L = 1.0, 5.0, and 10.0 Mbits

        :param waiting_policies: The collection of plot traces for drawing the optimal waiting state policy plot
                                 [Concurrency Op]
        """
        p_len, s_wait, o_star = (self.packet_length / 1e6), self.waiting_states, self.o_star

        with lock:
            waiting_policies.append(graph_objs.Scatter(x=s_wait, y=o_star, mode=self.PLOTLY_LINES_MARKERS_MODE,
                                                       name='Optimal Waiting State Policy: ' + r'$L{=}$' +
                                                            p_len + ' Mbits'))

        # Nothing to return

    # Core Visualization Routine
    def prepare_delay_power_tradeoff_data(self, trade_offs):
        """
        Expected Average Delay ($\Bar{T}$ s) v UAV Average Power Constraint ($P_{\text{avg}}$ W) w.r.t the optimal
        policy, for L = 1.0, 5.0, and 10.0 Mbits

        :param trade_offs: The collection of plot traces for illustrating the delay-power trade-off in comm states
                           [Concurrency Op]
        """
        p_len, pwr, t = self.packet_length, self.average_power_constraint, self.canary_comm_average_delay

        with lock:
            trace = trade_offs.get(p_len, graph_objs.Scatter(x=[], y=[], mode=self.PLOTLY_LINES_MARKERS_MODE,
                                                             name='Delay-Power TradeOff: ' + r'L{=}$' +
                                                                  p_len + ' Mbits'))
            x, y = list(trace.x), list(trace.y)
            x.append(pwr)
            y.append(t)
            trace.x, trace.y = x, y

        # Nothing to return...

    # Core Visualization Routine
    def prepare_no_relay_gns_data(self):
        """
        Map of GNs ($x$ m v $y$ m) that transmit directly to the BS w.r.t the optimal policy, for L = 1.0, 5.0, and
        10.0 Mbits

        :return Plotly data traces constituting a map of Ground Nodes (GNs) that directly transmit to the BS (and the
                ones that use the UAV relay) w.r.t the specified canary comm state
        """
        p_len, canary_state, plot_data = self.packet_length, self.comm_states[self.canary_state_index], list()
        indices = tf.where(tf.reduce_all(tf.equal(self.comm_states[:, 0], canary_state[0]), axis=1))
        states = tf.gather_nd(self.comm_states, indices, axis=0)
        xi_s = tf.gather(self.relay_status, indices, axis=0)
        gn_0 = tf.gather_nd(states, tf.where(tf.equal(xi_s, 0)), axis=0)[:, 1]

        # gn_1 = tf.gather_nd(states, tf.where(tf.not_equal(xi_s, 0)), axis=0)[:, 1]

        # Plot UAV position
        plot_data.append(graph_objs.Scatter(x=canary_state[0, 0], y=canary_state[0, 0], mode=self.PLOTLY_MARKERS_MODE,
                                            marker=dict(symbol=raw_symbols[0], color=raw_colors[0], size=30),
                                            name=r'UAV Position $(x_{U}, y_{U}){=}$' + canary_state[0].numpy()))

        # Plot GNs using Direct BS communication
        plot_data.append(graph_objs.Scatter(x=gn_0[:, 0], y=gn_0[:, 1], mode=self.PLOTLY_MARKERS_MODE,
                                            marker=dict(symbol=raw_symbols[1], color=raw_colors[1], size=20),
                                            name=r'GNs using Direct Base Station (BS) link for $L{=}$' +
                                                 p_len + ' Mbits'))

        # Plot GNs using the UAV-Relay
        # plot_data.append(graph_objs.Scatter(x=gn_1[:, 0], y=gn_1[:, 1], mode=self.PLOTLY_MARKERS_MODE,
        #                                     marker=dict(symbol=raw_symbols[2], color=raw_colors[2], size=20),
        #                                     name=r'GNs using the UAV Relay'))

        return plot_data

    # Core Visualization Routine
    def prepare_uav_trajectories_data(self):
        """
        The UAV trajectories ($x$ m v $y$ m) for randomly chosen GNs w.r.t the optimal policy, for L = 1.0, 5.0, and
        10.0 Mbits

        :return Plotly data traces constituting the UAV's position and GN positions w.r.t the specified canary comm
                state, along with optimal trajectories to address comm requests originating at these GNs
        """
        i, n_gns, idx = 0, self.NUMBER_OF_GROUND_NODES_FOR_TRAJECTORY_VISUALIZATION, self.canary_state_index
        canary_state, plot_data = self.comm_states[idx], list()
        indices = tf.where(tf.reduce_all(tf.equal(self.comm_states[:, 0], canary_state[0]), axis=1))[0:(n_gns - 1), :]
        states = tf.gather_nd(self.comm_states, indices, axis=0)[0:(n_gns - 1), :]

        # Plot UAV position
        plot_data.append(graph_objs.Scatter(x=canary_state[0, 0], y=canary_state[0, 0], mode=self.PLOTLY_MARKERS_MODE,
                                            marker=dict(symbol=raw_symbols[0], color=raw_colors[0], size=30),
                                            name=r'UAV Initial Position $(x_{0}, y_{0}){=}$' + canary_state[0].numpy()))

        for state in states:
            i += 1  # GN identifier
            opt_traj, symbol, color = self.optimal_trajectories[indices[i]], raw_symbols[i], raw_colors[i]

            # Plot GN_{i} position
            plot_data.append(graph_objs.Scatter(x=state[1, 0], y=canary_state[1, 1], mode=self.PLOTLY_MARKERS_MODE,
                                                marker=dict(symbol=symbol, color=color, size=20),
                                                name=f'Ground Node {i}: ' + r'$(x_{G}, y_{G}){=}$' +
                                                     canary_state[0].numpy()))

            # Plot the associated optimal trajectory
            plot_data.append(graph_objs.Scatter(x=opt_traj[:, 0], y=opt_traj[:, 1], mode=self.PLOTLY_LINES_MARKERS_MODE,
                                                marker=dict(symbol=symbol, color=color, size=10),
                                                line=dict(dash='dash', color=color, width=3),
                                                name=f'UAV Optimal Trajectory to serve Ground Node {i}: ' +
                                                     r'$\mathbf{p}^{*}$'))

        return plot_data

    # Core Visualization Routine
    def prepare_lagrangian_convergence_data(self):
        """
        The convergence of the Lagrangian cost metrics in the comm states ($l_{\nu}^{*}(s;\hat{r}_{U}, 1)$ v Time [s])

        :return A plotly scatter object constituting the convergence of the Lagrangian cost metrics in the comm states
                ($l_{\nu}^{*}(s;\hat{r}_{U}, 1)$ v Time [s])
        """
        t, lagr = self.canary_state_comm_costs.keys(), self.canary_state_comm_costs.values()
        return graph_objs.Scatter(x=t, y=lagr, mode=self.PLOTLY_LINES_MARKERS_MODE)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        print('[INFO] [{}] SMDPEvaluation Termination: Tearing things down - Exception Type = {} | Exception Value = {}'
              ' | Traceback = {}'.format(self.id, exc_type, exc_val, exc_tb))

        # Nothing to do here...


"""
External Handling Ops
"""


def plot_waiting_state_policy(pwr, plot_traces):
    """
    Plot UAV Radial Velocities ($v_{r}$ m/s) v UAV Radius Levels ($r_{U}$ m) w.r.t the optimal waiting state policy,
    for L = 1.0, 5.0, and 10.0 Mbits (x1 plot)

    :param pwr: The [common] average power constraint (in W) corresponding to the evaluators which provided these
                data traces (for different packet lengths)
    :param plot_traces: The plot data ($v_{r}$ m/s v $r_{U}$ m) for different packet lengths passed to the Plotly API
    """
    pwr /= 1e3  # Convert to kW

    plot_layout = dict(title='The optimal waiting state policy (Radial Velocity v Radius Level) for different data '
                             'packet lengths, with P_avg = {} kW'.format(pwr),
                       xaxis=dict(title=r'UAV Radius Level $v_{r}$ (in m)'),
                       yaxis=dict(title=r'UAV Radial Velocity $r_{U}$ (in m/s)'))
    fig = dict(data=plot_traces, layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='Odin_UAV_Mobility_Optimal_Waiting_State_Policy.png', auto_open=False)

    print('[INFO] [Main Thread] SMDPEvaluation plot_waiting_state_policy: The plot of the optimal waiting state policy '
          'for different data packet lengths--with P_avg = {} kW, is available here - {}'.format(pwr, fig_url))

    # Nothing to return


def plot_delay_power_tradeoff(plot_traces_dict):
    """
    Plot the Expected Average Delay ($\Bar{T}$ s) v UAV Average Power Constraint ($P_{\text{avg}}$ W) w.r.t the optimal
    policy, for L = 1.0, 5.0, and 10.0 Mbits (multiple plots)

    :param plot_traces_dict: The plot data traces dictionary {L: ($\Bar{T}$ s v $P_{\text{avg}}$ W) for different
                             values of the data packet length ($L$), passed--post-processing--to the Plotly API
    """
    plot_layout = dict(title='The delay-power trade-off w.r.t the optimal policy '
                             '(Expected Average Delay v Average Power Consumption Constraint)',
                       xaxis=dict(title=r'Expected Average Delay $\Bar{T}$ (in s)'),
                       yaxis=dict(title=r'UAV Average Power Constraint $P_{\text{avg}}$ (in W)'))

    for plen, plot_trace in plot_traces_dict.items():
        plen /= 1e6  # Convert to Mbits
        fig = dict(data=[plot_trace], layout=plot_layout)
        fig_url = plotly.plotly.plot(fig, filename='Odin_UAV_Mobility_Delay_Power_TradeOff.png', auto_open=False)

        print('[INFO] [Main Thread] SMDPEvaluation plot_delay_power_tradeoff: The plot of the delay-power trade-off '
              'for data packet length L = {} Mbits, is available here - {}'.format(plen, fig_url))

    # Nothing to return


def plot_no_relay_gns(pwr, plot_traces):
    """
    Plot a Map of GNs ($x$ m v $y$ m) that transmit directly to the BS w.r.t the optimal policy, for L = 1.0, 5.0, and
    10.0 Mbits

    :param pwr: The [common] average power constraint (in W) corresponding to the evaluators which provided these
                data traces (for different packet lengths)
    :param plot_traces: The plot data ($x$ m v $y$ m) for different packet lengths passed to the Plotly API
    """
    pwr /= 1e3  # Convert to kW

    plot_layout = dict(title='A Map of Ground Nodes (GNs) that transmit directly to the Base Station (BS) for '
                             'different data packet lengths, with P_avg = {} kW'.format(pwr),
                       xaxis=dict(title=r'$x$ (in m)'),
                       yaxis=dict(title=r'$y$ (in m)'))
    fig = dict(data=plot_traces, layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='Odin_UAV_Mobility_No_Relay_GNs_Map.png', auto_open=False)

    print('[INFO] [Main Thread] SMDPEvaluation plot_no_relay_gns: A map of Ground Nodes (GNs) that transmit directly '
          'to the Base Station (BS) under different data packet lengths--with P_avg = {} kW, '
          'is available here - {}'.format(pwr, fig_url))

    # Nothing to return


def plot_uav_trajectories(canary, plen, pwr, plot_trace):
    """
    Plot the UAV trajectories ($x$ m v $y$ m) for randomly chosen GNs w.r.t the optimal policy

    :param canary: The initial UAV position (in rectangular coordinates) is encapsulated in this tensor
    :param plen: The packet length (in bits) corresponding to the evaluator which provided this data trace
    :param pwr: The average power constraint corresponding to the evaluator which provided this trace data
    :param plot_trace: The plot data ($x$ m v $y$ m) passed to the Plotly API
    """
    plen /= 1e6  # Convert to Mbits
    pwr /= 1e3  # Convert to kW

    plot_layout = dict(title='The optimal HCSO-determined UAV trajectories for specific Ground Nodes (GNs), with data '
                             'packet length L = {} Mbits and P_avg = {} kW'.format(plen, pwr),
                       xaxis=dict(title=r'$x$ (in m)'),
                       yaxis=dict(title=r'$y$ (in m)'))
    fig = dict(data=[plot_trace], layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='Odin_UAV_Mobility_HCSO_UAV_Trajectories.png', auto_open=False)

    print('[INFO] [Main Thread] SMDPEvaluation plot_uav_trajectories: The plot of the optimal HCSO-determined UAV '
          'trajectories for the given Ground Nodes (GNs)--with L = {} Mbits | P_avg = {} kW | '
          'Initial UAV Position = {} [m, m], is available here - {}'.format(plen, pwr, canary[0], fig_url))

    # Nothing to return


def plot_lagrangian_convergence(canary, plen, pwr, plot_trace):
    """
    Plot the Lagrangian cost convergence for the given canary state ($l_{\nu}^{*}(s;\hat{r}_{U}, 1)$ v Time [s])

    :param canary: The canary comm state whose Lagrangian cost convergence is to be plotted
    :param plen: The packet length corresponding to the evaluator which provided this trace data
    :param pwr: The average power constraint corresponding to the evaluator which provided this trace data
    :param plot_trace: The plot data ($l_{\nu}^{*}(s;\hat{r}_{U}, 1)$ v Time [s]) passed to the Plotly API
    """
    plen /= 1e6  # Convert to Mbits
    pwr /= 1e3  # Convert to kW

    plot_layout = dict(title='Lagrangian Cost Convergence for a canary comm state, with data packet length '
                             'L = {} Mbits and P_avg = {} kW'.format(plen, pwr),
                       xaxis=dict(title='Time (in s)'),
                       yaxis=dict(title=r'$(1{-}\nu P_{\text{avg}})\Delta + \nu E$ (in J)'))
    fig = dict(data=[plot_trace], layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='Odin_UAV_Mobility_Lagrangian_Cost_Convergence.png', auto_open=False)

    print('[INFO] [Main Thread] SMDPEvaluation plot_cost_convergence: The plot of the Lagrangian Cost metric versus '
          'Computation Time for the communication state s = {}--with L = {} Mbits | P_avg = {} kW, '
          'is available here - {}'.format(canary, plen, pwr, fig_url))

    # Nothing to return


def launch_evaluation(id_, power_constraint, packet_len_constraint, canary_index, evaluators_):
    """
    Launch an SMDPEvaluation instance given the constraints in order to analyze its performance and functionalities

    :param id_: A unique identifier for the evaluator/agent that is to be launched from this routine
    :param power_constraint: The UAV average power consumption constraint for an SMDPEvaluation instance
    :param packet_len_constraint: The data packet length constraint for an SMDPEvaluation instance
    :param canary_index: The index of the canary state (in the comm state space) for visualizations
    :param evaluators_: A collection to house the evaluator instances for post-processing
    """
    with SMDPEvaluation(id_, power_constraint, packet_len_constraint, canary_index, evaluators_) as evaluator__:
        evaluator__.projected_subgradient_ascent()

    # Nothing to return...


# Run Trigger
if __name__ == '__main__':
    print('[INFO] [Main Thread] SMDPEvaluation main: Starting the evaluation of the proposed SMDP - HCSO formulation '
          'for adaptive multi-scale scheduling and trajectory optimization for power-constrained UAV relays...')

    evaluators = list()  # A collection to house the evaluator instances for post-processing

    pwrs, wait_policies, gn_maps, t_p_tradeoffs = list(), list(), list(), dict()  # Individual Plots & Data Collections

    canary_idx = 30  # The index of the canary state (in the comm state space) for visualizations

    # --- Values used in the JSAC SI 2020 paper ---
    # Average UAV Power Consumption Constraint (W)
    # avg_powers = np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)
    # Average Packet Length Constraint (bits)
    # packet_lens = np.array([1e6, 5e6, 10e6])

    avg_powers, packet_lens = np.array([1.1e3]), np.array([5e6])  # Test samples
    max_workers = (2000 * avg_powers.shape[0] * packet_lens.shape[0])

    with ThreadPoolExecutor(max_workers=max_workers) as exxeggutor:
        for packet_len in packet_lens:
            for avg_power in avg_powers:
                exxeggutor.submit(launch_evaluation, uuid.uuid4(), avg_power, packet_len, canary_idx, evaluators)

    with ThreadPoolExecutor(max_workers=max_workers) as exxeggutor:
        for evaluator in evaluators:
            len__, avg_p = evaluator.data_payload_size, evaluator.average_power_constraint

            # --- Sequencing according to the order in the JSAC SI 2020 paper ---
            exxeggutor.submit(evaluator.prepare_delay_power_tradeoff_data, t_p_tradeoffs)  # Delay-Power TradeOff data

            if avg_p == 1.2e3:
                exxeggutor.submit(evaluator.prepare_waiting_state_policy_data, wait_policies)  # Wait Policy Data

                exxeggutor.submit(evaluator.prepare_no_relay_gns_data, gn_maps)  # Direct BS GN data

                if len__ == 1e6:
                    c_state = evaluator.comm_states[evaluator.canary_state_index]

                    exxeggutor.submit(plot_uav_trajectories, c_state, len__, avg_p,
                                      evaluator.prepare_uav_trajectories_data())  # Fig. 3 (b)

                    exxeggutor.submit(plot_lagrangian_convergence, c_state, len__, avg_p,
                                      evaluator.prepare_lagrangian_convergence_data())  # Fig. 2 (a)

    # Combined Plots
    with ThreadPoolExecutor(max_workers=max_workers) as exxeggutor:
        exxeggutor.submit(plot_waiting_state_policy, 1.2e3, wait_policies)  # Fig. 2 (b)

        exxeggutor.submit(plot_delay_power_tradeoff, t_p_tradeoffs)  # Fig. 4 (a), (b), and (c)

        exxeggutor.submit(plot_no_relay_gns, 1.2e3, gn_maps)  # Fig. 3 (a)

# The evaluation ends here...
