"""
This script constitutes the set of operations involved in the visualization of data-payload-size-specific choropleth
maps illustrating the distribution of Ground Nodes in the circular cell of radius $a$ meters, with respect to MAESTRO's
optimal/augmented scheduling policy involving the Base Station and a Dynamic UAV.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import plotly
import warnings
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from numpy.random import uniform
from collections import namedtuple
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# User Credentials
plotly.tools.set_credentials_file(username='bkeshav1', api_key='wLYizSJgmQPTpcJ68Tva')

"""
CONFIGURATIONS-I: Deployment Model
"""

# Pi | NumPy random seed assignment
pi = np.pi
np.random.seed(6)

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1000.0

# The number of GN requests in this visualization | Doubles as the NUMBER_OF_GNs
NUMBER_OF_REQUESTS = 10000

"""
CONFIGURATIONS-II: Channel Model
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
CONFIGURATIONS-III: Design Model
"""

# The convergence confidence level for optimization algorithms in this framework
CONVERGENCE_CONFIDENCE = 5

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-5

"""
Node Deployments: GNs; BS; Dynamic UAV (initial position)
"""

r_gns, th_gns = uniform(0, CELL_RADIUS ** 2, NUMBER_OF_REQUESTS) ** 0.5, uniform(0, 2 * pi, NUMBER_OF_REQUESTS)
x_gns = tf.constant(list(zip(r_gns * np.cos(th_gns), r_gns * np.sin(th_gns))), dtype=tf.float64)

x_bs = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0], dtype=tf.float64), axis=0), multiples=[NUMBER_OF_REQUESTS, 1])
x_u = tf.tile(tf.expand_dims(tf.constant([0.0, 0.0], dtype=tf.float64), axis=0), multiples=[NUMBER_OF_REQUESTS, 1])

"""
The system-wide resource for [Link Performance] evaluation
"""


# noinspection PyMethodMayBeStatic
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

    def __f_z(self, z):
        """
        Calculate the value of $f(Z)$, i.e., the Shannon-Hartley Channel Capacity in terms of the variable $Z$

        :param z: The optimization variable in the re-formulated primal rate optimization problem

        :return: The value of the function $f(Z)$ in the re-formulated primal rate optimization problem
        """
        b = self.channel_bandwidth
        return b * np.log2(1 + (0.5 * (z ** 2)))

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

    def calculate_adapted_throughput(self, d, phi, r_los, r_nlos, num_workers):
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
                executor.submit(self.calculate_adapted_throughput, d, phi, r_los_s[i], r_nlos_s[i], num_workers)
        phi_degrees = (180.0 / np.pi) * phi_s
        p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
        p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)
        return r_los_s, r_nlos_s, tf.add(tf.multiply(p_los, r_los_s), tf.multiply(p_nlos, r_nlos_s))

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


def trajectory(x_u_, x_g_, x_b_):
    # TODO: Read the optimal policy log file and find p_star & v_star for this state
    return tf.concat([x_u_, x_g_, x_b_], axis=0)


def visualize(num_workers):
    data_lens = [0.1e6, 1e6, 10e6, 100e6, 1e9, 10e9]
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2

    # GN-to-BS links
    gb_xy_distances = tf.norm(tf.subtract(x_gns, x_bs), axis=1)
    gb_heights = tf.constant(abs(BASE_STATION_HEIGHT - 0.0), shape=gb_xy_distances.shape, dtype=tf.float64)
    gb_distances = tf.sqrt(tf.add(tf.square(gb_xy_distances), tf.square(gb_heights)))
    gb_angles = tf.asin(tf.divide(gb_heights, gb_distances))

    link_perf_evaluator = LinkPerformanceEvaluator(CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, LoS_PATH_LOSS_EXPONENT,
                                                   NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT,
                                                   LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
                                                   PROPAGATION_ENVIRONMENT_PARAMETER_1,
                                                   PROPAGATION_ENVIRONMENT_PARAMETER_2, CONVERGENCE_CONFIDENCE,
                                                   BISECTION_METHOD_TOLERANCE)

    # Link Performance Evaluations (BS and HAP)
    bs_delays = link_perf_evaluator.evaluate(gb_distances, gb_angles, data_lens, num_workers).average_delays

    # Link Performance Evaluations (Dynamic UAV)
    uav_delays = {d_len: [] for d_len in data_lens}
    for d_len in data_lens:
        bs_gns, uav_gns = [], []
        for k in range(NUMBER_OF_REQUESTS):
            x_g = x_gns[k]
            p = trajectory(tf.expand_dims(x_u[0], axis=0), tf.expand_dims(x_g, axis=0), tf.expand_dims(x_bs[0], axis=0))
            midpoint = int(p.shape[0] / 2)
            # Heuristic 1: Moves with max UAV velocity
            t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1), 55.0)
            # GN-to-UAV links
            r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
            h_u = tf.constant(UAV_HEIGHT, shape=r_gu.shape, dtype=tf.float64)
            d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
            phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()
            r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
            r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i__, (d_gu__, phi_gu__) in enumerate(zip(d_gu, phi_gu)):
                    executor.submit(link_perf_evaluator.calculate_adapted_throughput, d_gu__, phi_gu__,
                                    r_los_gu[i__], r_nlos_gu[i__], num_workers)
            phi_degrees_gu = (180.0 / np.pi) * phi_gu
            p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
            p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
            r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))
            t_p_1 = (d_len - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))) / r_bar_gu[-1]
            # Forward (UAV --> BS)
            r_ub = tf.norm(p[midpoint:], axis=1)
            h_ub = tf.constant(abs(UAV_HEIGHT - BASE_STATION_HEIGHT), shape=r_ub.shape, dtype=tf.float64)
            d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
            phi_ub = tf.asin(tf.divide(h_ub, d_ub))
            r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
            r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for i__, (d_ub__, phi_ub__) in enumerate(zip(d_ub, phi_ub)):
                    executor.submit(link_perf_evaluator.calculate_adapted_throughput, d_ub__, phi_ub__,
                                    r_los_ub[i__], r_nlos_ub[i__], num_workers)
            phi_degrees_ub = (180.0 / np.pi) * phi_ub
            p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
            p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
            r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))
            t_p_2 = (d_len - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))) / r_bar_ub[-1]
            uav_delays[d_len].append(tf.reduce_sum(t) + t_p_1 + t_p_2)
            if np.argmin([bs_delays[d_len][k], uav_delays[d_len][k]]) == 0:
                bs_gns.append(x_gns[k].numpy())
            else:
                uav_gns.append(x_gns[k].numpy())
        fig = dict(data=[go.Scatter(x=np.array(bs_gns)[:, 0], y=np.array(bs_gns)[:, 1],
                                    mode='markers', marker_symbol='circle', name=f'BS | {d_len / 1e6} Mb'),
                         go.Scatter(x=np.array(uav_gns)[:, 0], y=np.array(uav_gns)[:, 1],
                                    mode='markers', marker_symbol='star', name=f'UAV | {d_len / 1e6} Mb')],
                   layout=dict(title='GN Distribution Visualization'))
        plotly.plotly.plot(fig)


# Run Trigger
if __name__ == '__main__':
    visualize(1024)
