"""
This script evaluates the performance of the BS alone servicing GN requests.
This script evaluates the performance of stationary UAVs hovering at a fixed height of $H_{U}$ meters above the ground.
The key metric analyzed in this version (v21.11) of the script is:
    "UAV Average Power Constraint (in Watts) v GN Active Communication Request Delay (in seconds)"; for
        Payload Lengths ($L$) = 1.0 Mb, 10.0 Mb, and 100.0 Mb; for both [a BS serving the GNs alone] and for
        [Number of UAVs ($N_{U}$) = 1, 2, and 3] serving the GNs with BS only acting as the receiver.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

"""
v21.11: Reference Models for Performance Comparison with the SMDP-HCSO Optimal Policy
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import time
import numpy as np
import tensorflow as tf
# from threading import Lock
from collections import namedtuple
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
CONFIGURATIONS-I: Various Model Parameters
"""

np.random.seed(6)

"""
Deployment Model Parameters
"""
# The number of orthogonal channels at the BS ($N_{K}$)
NUMBER_OF_CHANNELS = 10

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1000.0

# The total number of GNs (implies communication requests) in the cell under analysis
NUMBER_OF_REQUESTS = 10000

# The number of communication requests originating in the cell per second ($\Lambda$) in requests/second
# 1.0 Mb: One request every minute | 10.0 Mb: One request every 5 minutes | 100.0 Mb: One request every 30 minutes
ARRIVAL_RATES = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.5555e-4}

"""
Channel Model Parameters
"""
# The reference SNR level at a link distance of 1-meter
REFERENCE_SNR_AT_1_METER = 1e4

# The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
LoS_PATH_LOSS_EXPONENT = 2.0

# The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
NLoS_PATH_LOSS_EXPONENT = 2.8

# The bandwidth of each orthogonal channel assigned to this application ($B$) in Hz
CHANNEL_BANDWIDTH = 5e6

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

"""
Design Parameters
"""
# The convergence confidence level for optimization algorithms in this framework
CONVERGENCE_CONFIDENCE = 10

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

"""
UAV Motion Model & Power Consumption Profile Parameters
"""
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

"""
The system-wide resource for [Link Performance] evaluation
"""


class LinkPerformanceEvaluator(object):
    """
    Link Performance Evaluator: Rate Adaptation via Bisection | Average Throughput and Average Delay Computation
    """

    def __init__(self, channel_bandwidth, reference_snr_at_1_meter, los_path_loss_exponent,
                 nlos_path_loss_exponent, nlos_attenuation_constant, los_rician_factor_1, los_rician_factor_2,
                 propagation_environment_parameter_1, propagation_environment_parameter_2,
                 convergence_confidence, bisection_method_tolerance):
        """
        The initialization sequence

        :param channel_bandwidth: The bandwidth of each orthogonal channel assigned to this application ($B$) in Hz
        :param reference_snr_at_1_meter: The reference SNR level at a link distance of 1-meter
        :param los_path_loss_exponent: The path-loss exponent for Line of Sight (LoS) links ($\alpha$)
        :param nlos_path_loss_exponent: The path-loss exponent for Non-Line of Sight (NLoS) links ($\tilde{\alpha}$)
        :param nlos_attenuation_constant: The additional NLoS attenuation factor ($\kappa$)
        :param los_rician_factor_1: The propagation environment dependent coefficient ($k_{1}$) for the LoS Rician link
        :param los_rician_factor_2: The propagation environment dependent coefficient ($k_{2}$) for the LoS Rician link
        :param propagation_environment_parameter_1: The ($z_{1}$) parameter for LoS/NLoS probability determination
        :param propagation_environment_parameter_2: The ($z_{2}$) parameter for LoS/NLoS probability determination
        :param convergence_confidence: The convergence confidence level for optimization algorithms in this framework
        :param bisection_method_tolerance: The tolerance value for the bisection method to find the optimal value of $Z$
        """
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
        self.evaluation_output = namedtuple('link_performance_evaluation_output',
                                            ['los_throughputs', 'nlos_throughputs', 'average_throughputs',
                                             'average_delays', 'aggregated_average_delay'])

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    # noinspection PyMethodMayBeStatic
    def __f_z(self, z):
        """
        Calculate the value of $f(Z)$, i.e., the Shannon-Hartley Channel Capacity in terms of the variable $Z$

        :param z: The optimization variable in the re-formulated primal rate optimization problem

        :return: The value of the function $f(Z)$ in the re-formulated primal rate optimization problem
        """
        b = self.channel_bandwidth
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
        :param y: An incomplete random variable rendition ($y$) for CDF evaluation of the non-central
                  chi-squared distribution, i.e., $x{=}y Z^{2}$
        :param low: The lower bound of the function's domain
        :param high: The upper bound of the function's domain
        :param tolerance: The tolerance level, which when achieved should terminate the optimization

        :return: The minimizer of the provided objective function, i.e., argmin
        """
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
        """
        Calculate the value of the optimization variable $Z$

        :param gamma: The data rate ($\mathbf{\gamma}$) of the Tx, employed in the calculation of $Z$

        :return: The value of the optimization variable
                 $Z{=}\sqrt{\frac{2 \beta P}{\sigma^{2} \Gamma} u(\mathbf{\gamma}, \beta)}$
        """
        b = self.channel_bandwidth
        return np.sqrt(2 * ((2 ** (gamma / b)) - 1))

    def __u(self, gamma, d, los):
        """
        Calculate the value of the function $u(\mathbf{\gamma}, \beta)$

        :param gamma: The data rate ($\mathbf{\gamma}$) of the Tx, employed to find $Z$ via $u(\mathbf{\gamma}, \beta)$
        :param d: The distance between the Tx and Rx nodes for determining the large-scale channel variations term
                  $\beta$ for use in the calculation of this function $u(\mathbf{\gamma}, \beta)$
        :param los: This boolean member will be True if the calculation of this function is for LoS settings

        :return: The value of the function $u(\mathbf{\gamma}, \beta)$
        """
        b, gamma_ = self.channel_bandwidth, self.reference_snr_at_1_meter
        alpha, alpha_, kappa = self.los_path_loss_exponent, self.nlos_path_loss_exponent, self.nlos_attenuation_constant
        return ((2 ** (gamma / b)) - 1) / (gamma_ * (lambda: kappa, lambda: 1)[los]() *
                                           (d ** (lambda: -alpha_, lambda: -alpha)[los]()))

    def __evaluate_los_throughput(self, d, phi, r_los):
        """
        Calculate the contribution of LoS transmissions to the total link throughput

        :param d: The distance ($d$) between the Tx and the Rx for the uplink link under analysis
        :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
        :param r_los: The optimized LoS throughput member fed into the routine for reference-based concurrent updates
        """
        k_1, k_2 = self.los_rician_factor_1, self.los_rician_factor_2
        k, alpha = k_1 * np.exp(k_2 * phi), self.los_path_loss_exponent
        b, gamma_ = self.channel_bandwidth, self.reference_snr_at_1_meter
        df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), self.bisection_method_tolerance
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) *
                                                                            gamma_ * (d ** -alpha))), t)
        tf.compat.v1.assign(r_los, self.__f_z(z_star), use_locking=True)

    def __evaluate_nlos_throughput(self, d, r_nlos):
        """
        Calculate the contribution of NLoS transmissions to the total link throughput

        :param d: The distance ($d$) between the Tx and the Rx for the uplink link under analysis
        :param r_nlos: The optimized NLoS throughput member fed into the routine for reference-based concurrent updates
        """
        alpha_, t = self.nlos_path_loss_exponent, self.bisection_method_tolerance
        b, gamma_, kappa = self.channel_bandwidth, self.reference_snr_at_1_meter, self.nlos_attenuation_constant
        df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) *
                                                                            gamma_ * kappa * (d ** -alpha_))), t)
        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), use_locking=True)

    def __calculate_adapted_throughput(self, d, phi, r_los, r_nlos, num_workers):
        """
        Calculate the LoS and NLoS throughput members of the link under analysis

        :param d: The distance ($d$) between the Tx and the Rx for the uplink link under analysis
        :param phi: The elevation angle ($\phi$) between the Tx and the Rx for the uplink link under analysis
        :param r_los: The optimized LoS throughput member fed into the routine for reference-based concurrent updates
        :param r_nlos: The optimized NLoS throughput member fed into the routine for reference-based concurrent updates
        :param num_workers: The number of concurrent executors to be employed during this calculation
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.submit(self.__evaluate_los_throughput, d, phi, r_los)
            executor.submit(self.__evaluate_nlos_throughput, d, r_nlos)

    def __average_throughputs(self, d_s, phi_s, num_workers):
        """
        Calculate the average throughputs for the various Tx-Rx distance-varied/angle-varied links

        :param d_s: The tensor constituting the distances between the Tx and the Rx (whose position is varied)
        :param phi_s: The tensor constituting the angles between the Tx and the Rx (whose position is varied)
        :param num_workers: The number of concurrent executors to be employed during this calculation

        :return: The various throughput metrics for the various Tx-Rx distance-varied/angle-varied links
        """
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
        """
        Calculate the average delays for the various Tx-Rx distance-varied/angle-varied links (constituted in r_bar)
        and the different payload lengths under analysis (constituted in p_lens)

        :param p_lens: The different payload lengths under analysis in this evaluation ($L$s)
        :param r_bars: The average throughputs for the various Tx-Rx distance-varied/angle-varied links

        :return: A dictionary of the various delay metrics keyed w.r.t individual payload lengths
        """
        delta_bars_p = {
            p_len: tf.divide(tf.constant(p_len, shape=r_bars.shape, dtype=tf.float64), r_bars) for p_len in p_lens
        }
        delta_bars_agg = {
            p_len: tf.divide(tf.reduce_sum(delta_bars), tf.constant(r_bars.shape[0], dtype=tf.float64)).numpy()
            for p_len, delta_bars in delta_bars_p.items()
        }
        return delta_bars_p, delta_bars_agg

    def evaluate(self, d_s, phi_s, p_lens, num_workers):
        """
        Evaluate link performance for a given GN-distribution (d_s and phi_s) and for different payload lengths

        :param d_s: The tensor constituting the distances between the Tx and the Rx (whose position is varied)
        :param phi_s: The tensor constituting the angles between the Tx and the Rx (whose position is varied)
        :param p_lens: The different payload lengths under analysis in this evaluation ($L$s)
        :param num_workers: The number of concurrent executors to be employed during this evaluation

        :return: A populated instance of this evaluator's namedtuple output
        """
        r_los_s, r_nlos_s, r_bars = self.__average_throughputs(d_s, phi_s, num_workers)
        delta_bars_p, delta_bars_agg = self.__average_delays(p_lens, r_bars)
        return self.evaluation_output(los_throughputs=r_los_s, nlos_throughputs=r_nlos_s, average_throughputs=r_bars,
                                      average_delays=delta_bars_p, aggregated_average_delay=delta_bars_agg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit (if any)
        :param exc_val: The value or relevant data/information associated with the raised exit-exception (if any)
        :param exc_tb: The traceback details of the raised exit-exception (if any)
        """
        print(f'[INFO] LinkPerformanceEvaluator Termination: Tearing things down - Exception Type = {exc_type} | '
              f'Exception Value = {exc_val} | Traceback = {exc_tb}')


"""
The system-wide resource for [UAV Mobility & Power Consumption] evaluation
"""


def evaluate_power_consumption(uav_flying_velocity):
    """
    Determine the amount of power consumed by the UAV in Watts ($P_{\text{mob}}(v_{u}(t))$ W)

    :param uav_flying_velocity: The instantaneous horizontal flying velocity of the UAV in m/s ($v_{u}(t))

    :return: The UAV's power consumption when flying at the specified velocity (or hovering) according to the
             provided motion and power consumption profiles
    """
    v, u_tip, v_0 = uav_flying_velocity, ROTOR_BLADE_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY
    p_1, p_2, p_3 = POWER_PROFILE_CONSTANT_1, POWER_PROFILE_CONSTANT_2, POWER_PROFILE_CONSTANT_3
    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + \
           (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5) + \
           (p_3 * (v ** 3))


"""
The system-wide resource for [Poisson Arrivals] emulation [Custom]
"""

# def simulate_poisson_arrivals(payload_length):
#     """
#     Simulate the Poisson arrival process of the active communication requests originating from the GNs in the cell
#
#     :param: The data payload length input which serves as a key into the ARRIVAL_RATE dictionary in order to select
#             a payload-appropriate arrival rate
#
#     :return: A list of payload-size-specific GN active communication request arrival times
#     """
#     arrival_time, arrival_rate, arrival_times = 0.0, ARRIVAL_RATES[payload_length], []
#     for request_number in range(NUMBER_OF_REQUESTS):
#         arrival_time -= np.log(np.random.random_sample()) / arrival_rate
#         arrival_times.append(arrival_time)
#     return arrival_times


"""
The system-wide resources for [M/G/$N_{K}$] Queueing System evaluation [Custom]
"""

# sleep_seconds, number_of_requests, number_of_servers = 1e-9, NUMBER_OF_REQUESTS, NUMBER_OF_CHANNELS
# queue = tf.Variable(tf.zeros(shape=(number_of_requests,), dtype=tf.int8), dtype=tf.int8)
# wait_times = tf.Variable(tf.zeros(shape=(number_of_requests,), dtype=tf.float64), dtype=tf.float64)
# lock, server_pool = Lock(), tf.Variable(tf.zeros(shape=(number_of_servers,), dtype=tf.int8), dtype=tf.int8)


# def service_request(request_number, available_servers, service_times):
#     """
#     Service the GN request: Consume | Occupy a server | Sleep for the service period | Un-occupy the server
#
#     :param request_number: The GN request index for a global post-processing assignment of its service delay
#     :param available_servers: A tensor of available servers at this point in the emulation
#     :param service_times: The communication service delays for each GN request arriving at the serving node
#
#     :return: The server index (>0 | SERVED) or (-1 | NOT SERVED)
#     """
#     server = available_servers[0, 0].numpy() if tf.not_equal(tf.size(available_servers), 0) else -1
#     if server != -1:
#         tf.compat.v1.assign(queue[request_number], 1, use_locking=True)
#         tf.compat.v1.assign(server_pool[server], 1, use_locking=True)
#         time.sleep(service_times[request_number].numpy())
#         tf.compat.v1.assign(server_pool[server], 0, use_locking=True)
#     return server


# def life_of_a_request(start_time, request_number, call_number, service_times):
#     """
#     A thread-safe, executor-handled routine that emulates the life of a GN request
#
#     :param start_time: The time at which the processing of this GN request started
#     :param request_number: The GN request index for a global post-processing assignment of its service delay
#     :param call_number: This function call's place in the overall process hierarchy of the GN request under analysis
#     :param service_times: The communication service delays for each GN request arriving at the serving node
#
#     :return: A boolean indicating whether the request has been served
#     """
#     served = False
#     while not served:
#         with lock:
#             available_servers = tf.where(tf.equal(server_pool, 0))
#             queue_go_ahead = tf.equal(queue[request_number - 1], 1) \
#                 if request_number > 0 and call_number == 0 else True
#         if queue_go_ahead:
#             if service_request(request_number, available_servers, service_times) == -1:
#                 print(f'Call Number = {call_number} | Waiting {request_number}')
#                 time.sleep(sleep_seconds)
#                 served = life_of_a_request(start_time, request_number, call_number + 1, service_times)
#             else:
#                 print(f'Call Number = {call_number} | Served {request_number}!')
#                 tf.compat.v1.assign(wait_times[request_number], (time.time_ns() - start_time) / 1e9, use_locking=True)
#                 served = True
#         else:
#             time.sleep(sleep_seconds)
#             continue
#     return served


"""
The system-wide resources for [M/G/$N_{K}$] Queueing System evaluation [SimPy]
"""


def gn_request(env, num, chs, w_times, serv_times):
    """
    Process the generated GN request

    :param env: The SimPy environment wherein the queueing model is being evaluated
    :param num: The GN request index
    :param chs: The number of orthogonal BS channels (SimPy Resources) in this evaluation
    :param w_times: The wait times collection for logging the request queue wait times
    :param serv_times: The service-times collection housing the time taken to serve each GN request
    """
    arrival_time = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])
    with chs[k].request() as req:
        yield req
        w_times.append(env.now - arrival_time)
        yield env.timeout(serv_times[num])


def arrivals(env, chs, n_r, arr, w_times, serv_times):
    """
    Simulate Poisson arrivals and process the requests according to their service times

    :param env: The SimPy environment wherein the queueing model is being evaluated
    :param chs: The number of orthogonal BS channels (SimPy Resources) in this evaluation
    :param n_r: The total number of GN requests being analyzed in this queueing model evaluation
    :param arr: The arrival rate of GN requests for this Poisson simulation (payload-length specific rate)
    :param w_times: The wait times collection for logging the request queue wait times
    :param serv_times: The service-times collection housing the time taken to serve each GN request
    """
    for num in range(n_r):
        env.process(gn_request(env, num, chs, w_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


"""
Reference Models
"""


def no_uav_relay(payload_lengths, bs_coords, gn_coords, num_workers):
    """
    Evaluate the average delay performance of the cell when there is no UAV relay

    :param payload_lengths: The data payload lengths for which our framework's operations have to be evaluated
    :param bs_coords: The coordinates of the Base Station in the cell under analysis
    :param gn_coords: The coordinates (distance- and angle-varied) of the Ground Nodes in the cell under analysis
    :param num_workers: The number of concurrent executors to be employed during this evaluation

    :return: The average communication delays for the GN-to-BS links (as the distances and angles are varied)
    """
    # GN-to-BS links
    gb_xy_distances = tf.norm(tf.subtract(gn_coords, bs_coords), axis=1)
    gb_heights = tf.constant(abs(BASE_STATION_HEIGHT - 0.0), shape=gb_xy_distances.shape, dtype=tf.float64)
    gb_distances = tf.sqrt(tf.add(tf.square(gb_xy_distances), tf.square(gb_heights)))
    gb_angles = tf.asin(tf.divide(gb_heights, gb_distances))
    # Link Performance Evaluations
    with LinkPerformanceEvaluator(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, LoS_PATH_LOSS_EXPONENT,
                                  NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT,
                                  LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2, PROPAGATION_ENVIRONMENT_PARAMETER_1,
                                  PROPAGATION_ENVIRONMENT_PARAMETER_2, CONVERGENCE_CONFIDENCE,
                                  BISECTION_METHOD_TOLERANCE) as link_perf_evaluator:
        gb_delays = link_perf_evaluator.evaluate(gb_distances, gb_angles, payload_lengths, num_workers).average_delays
    return gb_delays


def single_uav_relay(payload_lengths, bs_coords, uav_coords, gn_coords, num_workers):
    """
    Evaluate the average delay performance of the cell when there is only one UAV serving as the BS-relay

    :param payload_lengths: The data payload lengths for which our framework's operations have to be evaluated
    :param bs_coords: The coordinates of the Base Station in the cell under analysis
    :param uav_coords: The coordinates of the stationary UAV relay in the cell under analysis
    :param gn_coords: The coordinates (distance- and angle-varied) of the Ground Nodes in the cell under analysis
    :param num_workers: The number of concurrent executors to be employed during this evaluation

    :return: The average communication delays for the [GN-to-UAV + UAV-to-BS] links
             (as the GN distances and angles are varied)
    """
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
    # Link Performance Evaluations
    with LinkPerformanceEvaluator(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, LoS_PATH_LOSS_EXPONENT,
                                  NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT,
                                  LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2, PROPAGATION_ENVIRONMENT_PARAMETER_1,
                                  PROPAGATION_ENVIRONMENT_PARAMETER_2, CONVERGENCE_CONFIDENCE,
                                  BISECTION_METHOD_TOLERANCE) as link_perf_evaluator:
        gu_delays = link_perf_evaluator.evaluate(gu_distances, gu_angles, payload_lengths, num_workers).average_delays
        ub_delays = link_perf_evaluator.evaluate(ub_distances, ub_angles, payload_lengths, num_workers).average_delays
    return {p_len: tf.add(gu_delays[p_len], ub_delays[p_len]) for p_len in payload_lengths}


def multiple_uav_relays(payload_lengths, bs_coords, multiple_uav_coords, gn_coords, num_workers):
    """
    Evaluate the average delay performance of the cell when there are multiple UAVs serving as BS-relays

    :param payload_lengths: The data payload lengths for which our framework's operations have to be evaluated
    :param bs_coords: The coordinates of the Base Station in the cell under analysis
    :param multiple_uav_coords: The coordinates of the multiple stationary UAV relays in the cell under analysis
    :param gn_coords: The coordinates (distance- and angle-varied) of the Ground Nodes in the cell under analysis
    :param num_workers: The number of concurrent executors to be employed during this evaluation

    :return: The average communication delays for the [GN-to-UAV + UAV-to-BS] links w.r.t multiple UAV relays
             (as the GN distances and angles are varied)
    """
    # GN-to-UAV links
    gu_xy_distances = [tf.norm(tf.subtract(gn_coords, uav_coords), axis=1) for uav_coords in multiple_uav_coords]
    gu_heights = tf.constant(abs(UAV_HEIGHT - 0.0), shape=gu_xy_distances[0].shape, dtype=tf.float64)
    gu_distances = tf.concat([tf.expand_dims(tf.sqrt(tf.add(tf.square(gu_xy_dists), tf.square(gu_heights))), axis=1)
                              for gu_xy_dists in gu_xy_distances], axis=1)
    gu_angles = tf.concat([tf.expand_dims(tf.asin(tf.divide(gu_heights, gu_distances[:, idx])), axis=1)
                           for idx in range(len(gu_xy_distances))], axis=1)
    min_indices = tf.math.argmin(gu_distances, axis=1)
    gu_distances_min = tf.reduce_min(gu_distances, axis=1)
    gu_angles_min = tf.constant([gu_angles[i, min_indices[i]].numpy() for i in range(min_indices.shape[0])],
                                dtype=tf.float64)
    # UAV-to-BS links
    ub_xy_distances = tf.constant([tf.norm(tf.subtract(multiple_uav_coords[i], bs_coords), axis=1)[0].numpy()
                                   for i in min_indices.numpy()], dtype=tf.float64)
    ub_heights = tf.constant(abs(UAV_HEIGHT - BASE_STATION_HEIGHT), shape=ub_xy_distances.shape, dtype=tf.float64)
    ub_distances_min = tf.sqrt(tf.add(tf.square(ub_xy_distances), tf.square(ub_heights)))
    ub_angles_min = tf.asin(tf.divide(ub_heights, ub_distances_min))
    # Link Performance Evaluations
    with LinkPerformanceEvaluator(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, LoS_PATH_LOSS_EXPONENT,
                                  NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT,
                                  LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2, PROPAGATION_ENVIRONMENT_PARAMETER_1,
                                  PROPAGATION_ENVIRONMENT_PARAMETER_2, CONVERGENCE_CONFIDENCE,
                                  BISECTION_METHOD_TOLERANCE) as link_perf_evaluator:
        gu_delays = link_perf_evaluator.evaluate(gu_distances_min, gu_angles_min,
                                                 payload_lengths, num_workers).average_delays
        ub_delays = link_perf_evaluator.evaluate(ub_distances_min, ub_angles_min,
                                                 payload_lengths, num_workers).average_delays
    return {p_len: tf.add(gu_delays[p_len], ub_delays[p_len]) for p_len in payload_lengths}
    # return min_indices, {p_len: tf.add(gu_delays[p_len], ub_delays[p_len]) for p_len in payload_lengths}


"""
CONFIGURATIONS-II: Constant that were not defined at the top of this script can be configured in this routine
"""


def simulate_operations(num_workers):
    """
    Simulate the operations of our framework

    :param num_workers: The number of concurrent executors to be employed during this evaluation
    """
    number_of_gn_requests, payload_lengths, gn_coords = NUMBER_OF_REQUESTS, [1e6, 10e6, 100e6], []
    bs_coords = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0], dtype=tf.float64), axis=0),
                        multiples=[number_of_gn_requests, 1])

    # arrival_times_dict = {p_len: simulate_poisson_arrivals(p_len) for p_len in payload_lengths}

    # Uniformly-random GN position generation heuristic

    radii = np.random.uniform(0, CELL_RADIUS ** 2, number_of_gn_requests) ** 0.5
    angles = np.random.uniform(0, 2 * np.pi, number_of_gn_requests)
    gn_coords = tf.constant(list(zip(radii * np.cos(angles), radii * np.sin(angles))), dtype=tf.float64)

    # Reference-1: No UAV Relay | M/G/1 Queueing System [SimPy Queueing Emulation]
    comm_delays_dict = no_uav_relay(payload_lengths, bs_coords, gn_coords, num_workers)

    # Reference-3: No UAV Relay | $M/G/N_{K}$ Queueing System [SimPy Queueing Emulation]
    number_of_servers = NUMBER_OF_CHANNELS
    for payload_length, comm_delays_tensor in comm_delays_dict.items():
        waiting_times, comm_delays, environment = [], comm_delays_tensor.numpy(), Environment()
        environment.process(arrivals(environment, [Resource(environment) for _ in range(number_of_servers)],
                                     number_of_gn_requests, ARRIVAL_RATES[payload_length], waiting_times, comm_delays))
        environment.run()
        print(f'[DEBUG] ReferenceModels simulate_operations: Payload Size = {payload_length / 1e6} Mb | '
              f'Average Comm Delay = {tf.reduce_mean(comm_delays)} seconds')
        print(f'[DEBUG] ReferenceModels simulate_operations: Payload Size = {payload_length / 1e6} Mb | '
              f'Average Wait Delay = {np.mean(waiting_times)} seconds')
        print(f'[INFO] ReferenceModels simulate_operations: No UAV Relay | M/G/{number_of_servers} | '
              f'All requests are served by the BS | Payload Length = [{payload_length / 1e6}] Mb | '
              f'UAV Power Consumption Constraint = [N/A] Watts | '
              f'Average Service Delay = '
              f'{tf.reduce_mean(tf.add(tf.constant(waiting_times, dtype=tf.float64), comm_delays))} seconds\n')


# Run Trigger
if __name__ == '__main__':
    simulate_operations(num_workers=256)
