"""
This script constitutes the evaluation of the channel model employed in our Phase-I UAV fleet automation engine.

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

"""
v21.11: Standalone Model Evaluation
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logging level

import plotly
import numpy as np
import tensorflow as tf
import plotly.graph_objs as graph_objs
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

plotly.tools.set_credentials_file(username='bkeshav1', api_key='wLYizSJgmQPTpcJ68Tva')  # Plotly User Credentials

"""
CONFIGURATIONS-I: Various Model Parameters
"""

"""
Deployment Model Parameters
"""
# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1000.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 60.0

# The number of radius levels ($r$) in the cell under analysis
NUMBER_OF_RADIUS_LEVELS = 1000

"""
Channel Model Parameters
"""
# The bandwidth of each orthogonal channel assigned to this application ($B$) in Hz
CHANNEL_BANDWIDTH = 1e6

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
#   Note that this is $\frac{\beta_{0} P}{\sigma^{2} \Gamma}{=}40 \text{dB}$
REFERENCE_SNR_AT_1_METER = 1e4

# The propagation environment specific parameter ($z_{1}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_1 = 9.61

# The propagation environment specific parameter ($z_{2}$) for LoS/NLoS probability determination
PROPAGATION_ENVIRONMENT_PARAMETER_2 = 0.16

"""
Design Parameters
"""
# The convergence confidence level for optimization algorithms in this framework
CONVERGENCE_CONFIDENCE = 100

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

"""
Visualization Parameters
"""
# The Plotly API "markers-only" scatter plot mode
PLOTLY_MARKERS_MODE = 'markers'

# The Plotly API "lines and markers" scatter plot mode
PLOTLY_LINES_MARKERS_MODE = 'lines+markers'

"""
Core Routines for Channel Model Evaluation
"""


def __f_z(z):
    """
    Calculate the value of $f(Z)$, i.e., the Shannon-Hartley Channel Capacity in terms of the variable $Z$

    :param z: The optimization variable in the re-formulated primal rate optimization problem

    :return: The value of the function $f(Z)$ in the re-formulated primal rate optimization problem
    """
    b = CHANNEL_BANDWIDTH
    return b * np.log2(1 + (0.5 * (z ** 2)))


def __marcum_q(df, nc, x):
    """
    Calculate the value of the Marcum-Q function using the specified values of the number of degrees of freedom $k$,
    the non-centrality factor $\lambda$, and the random variable rendition $x$

    :param df: The number of degrees of freedom ($k$) for this non-central chi-squared distribution
    :param nc: The non-centrality parameter ($\lambda$) for this non-central chi-squared distribution
    :param x: The random variable rendition ($x$) for CDF evaluation of this non-central chi-squared distribution

    :return: The value of the Marcum-Q function (1 - CDF[non-central chi-squared])
    """
    return 1 - ncx2.cdf(x, df, nc)


def __f(z, *args):
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
    f_z, q_m = __f_z(z), __marcum_q(df, nc, (y * (z ** 2)))
    ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
    ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf
    return -ln_f_z - ln_q_m


def __bisect(f, df, nc, y, low, high, tolerance):
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
    mid, converged, conf, conf_th = 0.0, False, 0, CONVERGENCE_CONFIDENCE
    while (not converged) or (conf < conf_th):
        mid = (high + low) / 2
        if (f(low, *args) * f(high, *args)) > 0.0:
            low = mid
        else:
            high = mid
        converged = (abs(high - low) < tolerance)
        conf += 1 if converged else -conf  # Update OR Reset if it is a red herring
    return mid


def __z(gamma):
    """
    Calculate the value of the optimization variable $Z$

    :param gamma: The data rate ($\mathbf{\gamma}$) of Tx, employed in the calculation of $Z$

    :return: The value of the optimization variable
             $Z{=}\sqrt{\frac{2 \beta P}{\sigma^{2} \Gamma} u(\mathbf{\gamma}, \beta)}$
    """
    b = CHANNEL_BANDWIDTH
    return np.sqrt(2 * ((2 ** (gamma / b)) - 1))


def __u(gamma, d, los):
    """
    Calculate the value of the function $u(\mathbf{\gamma}, \beta)$

    :param gamma: The data rate ($\mathbf{\gamma}$) of Tx, employed to find $Z$ through $u(\mathbf{\gamma}, \beta)$
    :param d: The distance between the Tx and Rx nodes for determining the large-scale channel variations term
              $\beta$ for use in the calculation of this function $u(\mathbf{\gamma}, \beta)$
    :param los: This boolean member will be True if the calculation of this function is for LoS settings

    :return: The value of the function $u(\mathbf{\gamma}, \beta)$
    """
    # Note that 'gamma_' is $\frac{\beta_{0} P}{\sigma^{2} \Gamma}{=}40 \text{dB}$ as indicated in Matt's manuscript
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    alpha, alpha_, kappa = LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT
    return ((2 ** (gamma / b)) - 1) / (gamma_ * (lambda: kappa, lambda: 1)[los]() *
                                       (d ** (lambda: -alpha_, lambda: -alpha)[los]()))


def __evaluate_los_throughput(d, phi, r_star_los):
    """
    Calculate the contribution of LoS transmissions to the total link throughput

    :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
    :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                (GU, UB, GB)
    :param r_star_los: The supplementary optimized LoS throughput member for post-evaluation visualization -- fed
                       into the routine for reference-based concurrent updates
    """
    k_1, k_2 = LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    k, alpha = k_1 * np.exp(k_2 * phi), LoS_PATH_LOSS_EXPONENT
    df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), BISECTION_METHOD_TOLERANCE
    z_star = __bisect(__f, df, nc, y, 0, __z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ *
                                                         (d ** -alpha))), t)
    gamma_star = __f_z(z_star)
    tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)


def __evaluate_nlos_throughput(d, r_star_nlos):
    """
    Calculate the contribution of NLoS transmissions to the total link throughput

    :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
    :param r_star_nlos: The supplementary optimized NLoS throughput member for post-evaluation visualization -- fed
                        into the routine for reference-based concurrent updates
    """
    alpha_, t = NLoS_PATH_LOSS_EXPONENT, BISECTION_METHOD_TOLERANCE
    b, gamma_, kappa = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, NLoS_ATTENUATION_CONSTANT
    df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))
    z_star = __bisect(__f, df, nc, y, 0, __z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa *
                                                         (d ** -alpha_))), t)
    gamma_star = __f_z(z_star)
    tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)


def __calculate_adapted_throughput(d, phi, r_star_los, r_star_nlos):
    """
    Calculate the throughput of the link under analysis (GU, UB, GB) (LoS/NLoS) post rate-adaptation in $Z$

    :param d: The distance ($d$) between the Tx and Rx nodes for the uplink link under analysis (GU, UB, GB)
    :param phi: The elevation angle ($\phi$) between the Tx and Rx nodes for the uplink link under analysis
                (GU, UB, GB)
    :param r_star_los: The supplementary optimized LoS throughput member for post-evaluation visualization -- fed into
                       the routine for reference-based concurrent updates
    :param r_star_nlos: The supplementary optimized NLoS throughput member for post-evaluation visualization -- fed
                        into the routine for reference-based concurrent updates

    :return: The rate-adapted throughput for the uplink link under analysis (GU, UB, GB) (LoS/NLoS)
    """
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.submit(__evaluate_los_throughput, d, phi, r_star_los)
        executor.submit(__evaluate_nlos_throughput, d, r_star_nlos)


# Run Trigger
if __name__ == '__main__':
    """
    Setup-I
    """
    # The re-casted parameter variables
    bs_height, uav_height = BASE_STATION_HEIGHT, UAV_HEIGHT
    cell_radius, num_levels = CELL_RADIUS, NUMBER_OF_RADIUS_LEVELS
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2
    # Node (GN/UAV) position generation
    levels = np.linspace(start=100.0, stop=cell_radius, num=num_levels, dtype=np.float64)
    angle_choice = np.random.choice(np.linspace(start=0.0, stop=2 * np.pi, num=10 * num_levels))
    nodes = [_ * np.squeeze(np.einsum('ji', np.vstack([np.cos(angle_choice), np.sin(angle_choice)]))) for _ in levels]

    """
    CONFIGURATIONS-II: Reference Point Specifications
    """
    reference_coords, reference_height, node_height, link_name = [0.0, 0.0], bs_height, 0.0, 'GB'
    # reference_coords, reference_height, node_height, link_name = [0.0, 0.0], bs_height, uav_height, 'UB'
    # reference_coords, reference_height, node_height, link_name = [0.0, 0.0], uav_height, 0.0, 'GU'

    """
    Setup-II
    """
    # Distances and Angles evaluation: GN-BS links | GN-UAV links | BS-UAV links
    xy_distances = tf.norm(tf.subtract(nodes, reference_coords), axis=1)
    relative_heights = tf.constant(abs(reference_height - node_height), shape=xy_distances.shape, dtype=tf.float64)
    distances = tf.sqrt(tf.add(tf.square(xy_distances), tf.square(relative_heights))).numpy()
    angles = tf.asin(tf.divide(relative_heights, distances)).numpy()
    # The outputs for visualization
    los_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)
    nlos_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)

    """
    Channel Model evaluation
    """
    print(f'[INFO] [Main Thread] ChannelModel main: Starting the channel model [{link_name}] evaluations...')
    phi_degrees = (180.0 / np.pi) * angles
    r_los, r_nlos = los_throughputs, nlos_throughputs
    p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
    p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)
    with ThreadPoolExecutor(max_workers=8) as exeggutor:
        for index, (distance, angle) in enumerate(zip(distances, angles)):
            exeggutor.submit(__calculate_adapted_throughput, distance, angle, r_los[index], r_nlos[index])

    """
    Visualizations
    """
    r_avg = tf.add(tf.multiply(p_los, r_los), tf.multiply(p_nlos, r_nlos))
    delay_p1Mb = tf.constant(1e5, dtype=tf.float64) / r_avg
    delay_1Mb = tf.constant(1e6, dtype=tf.float64) / r_avg
    delay_5Mb = tf.constant(5e6, dtype=tf.float64) / r_avg
    x_vals, y_los, y_nlos, y_avg = distances, r_los.numpy() / 1e6, r_nlos.numpy() / 1e3, r_avg.numpy() / 1e6
    # LoS Throughput
    plot_trace_los = graph_objs.Scatter(x=x_vals, y=y_los, mode='lines+markers')
    plot_layout_los = dict(title=f'Channel Model Evaluations [{link_name}] | LoS links',
                           xaxis=dict(title='Tx-Rx Distance in m', autorange=True),
                           yaxis=dict(title=f'Optimized {link_name} LoS Throughput in Mbps',
                                      type='log', autorange=True))
    fig_los = dict(data=[plot_trace_los], layout=plot_layout_los)
    fig_los_url = plotly.plotly.plot(fig_los, filename=''.join(['Odin_', link_name, 'LoS_Link_Throughput_Analysis']),
                                     auto_open=False)
    print(f'[INFO] [Main Thread] ChannelModel main: The Distance v Throughput visualization assuming purely LoS links '
          f'for [{link_name}] is given here - [{fig_los_url}]')
    # NLoS Throughput
    plot_trace_nlos = graph_objs.Scatter(x=x_vals, y=y_nlos, mode='lines+markers')
    plot_layout_nlos = dict(title=f'Channel Model Evaluations [{link_name}] | NLoS links',
                            xaxis=dict(title='Tx-Rx Distance in m', autorange=True),
                            yaxis=dict(title=f'Optimized {link_name} NLoS Throughput in kbps',
                                       type='log', autorange=True))
    fig_nlos = dict(data=[plot_trace_nlos], layout=plot_layout_nlos)
    fig_nlos_url = plotly.plotly.plot(fig_nlos, filename=''.join(['Odin_', link_name, 'NLoS_Link_Throughput_Analysis']),
                                      auto_open=False)
    print(f'[INFO] [Main Thread] ChannelModel main: The Distance v Throughput visualization assuming purely NLoS links '
          f'for [{link_name}] is given here - [{fig_nlos_url}]')
    # Average Throughput with a probabilistic LoS/NLoS model
    plot_trace_avg = graph_objs.Scatter(x=x_vals, y=y_avg, mode='lines+markers')
    plot_layout_avg = dict(title=f'Channel Model Evaluations [{link_name}] | Average Throughput',
                           xaxis=dict(title='Tx-Rx Distance in m', autorange=True),
                           yaxis=dict(title=f'Optimized {link_name} Average Throughput in Mbps',
                                      type='log', autorange=True))
    fig_avg = dict(data=[plot_trace_avg], layout=plot_layout_avg)
    fig_avg_url = plotly.plotly.plot(fig_avg, filename=''.join(['Odin_', link_name, 'Average_Throughput_Analysis']),
                                     auto_open=False)
    print(f'[INFO] [Main Thread] ChannelModel main: The Distance v Average Throughput visualization assuming a LoS/NLoS'
          f' probabilistic occurrence model for link [{link_name}] is given here - [{fig_avg_url}]')
    # Average Delays $L{=}[0.1, 1.0, 5.0]$Mb
    plot_trace_delay_p1Mb = graph_objs.Scatter(x=x_vals, y=delay_p1Mb, name='0.1 Mb', mode='lines+markers')
    plot_trace_delay_1Mb = graph_objs.Scatter(x=x_vals, y=delay_1Mb, name='1.0 Mb', mode='lines+markers')
    plot_trace_delay_5Mb = graph_objs.Scatter(x=x_vals, y=delay_5Mb, name='5.0 Mb', mode='lines+markers')
    plot_layout_delay = dict(
        title=f'Channel Model Evaluations [{link_name}] | Average Delay for various packet lengths',
        xaxis=dict(title='Tx-Rx Distance in m', autorange=True),
        yaxis=dict(title=f'Average Delay in s [{link_name}]', autorange=True))
    fig_delay = dict(data=[plot_trace_delay_p1Mb, plot_trace_delay_1Mb, plot_trace_delay_5Mb], layout=plot_layout_delay)
    fig_delay_url = plotly.plotly.plot(fig_delay, filename=''.join(['Odin_', link_name, 'Average_Delay_Analysis']),
                                       auto_open=False)
    print(f'[INFO] [Main Thread] ChannelModel main: The Distance v Average Delay visualization assuming a LoS/NLoS'
          f' probabilistic occurrence model for link [{link_name}] is given here - [{fig_delay_url}]')

    """
    Teardown
    """
    print('[INFO] [Main Thread] Channel Model main: The channel model evaluations and associated visualizations '
          'have been completed.')
# The ChannelModel evaluation and associated visualizations end here...
