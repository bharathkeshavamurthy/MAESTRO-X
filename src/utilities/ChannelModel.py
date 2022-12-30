"""
This script constitutes the evaluation of the channel model employed in our Phase-I UAV fleet automation engine.

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

import plotly
import numpy as np
import tensorflow as tf
import plotly.graph_objs as graph_objs
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Miscellaneous
"""
np.random.seed(6)
plotly.tools.set_credentials_file(username='bkeshav1', api_key='BCvYNi3LNNXgfpDGEpo0')

"""
Configurations-III: Simulation parameters
"""

''' Deployment model '''

# The height of the BS from the ground ($H_{B}$) in meters
BASE_STATION_HEIGHT = 80.0

# The height of the UAV from the ground ($H_{U}$) in meters
UAV_HEIGHT = 200.0

# The number of radius levels ($r$) in the cell under analysis
NUMBER_OF_RADIUS_LEVELS = int(1e4)

# The radius of the circular cell under evaluation ($a$) in meters
CELL_RADIUS = 1e3

''' Channel model (urban radio environment) '''

# The additional NLoS attenuation factor ($\kappa$)
NLoS_ATTENUATION_CONSTANT = 0.2

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

# The reference SNR level at a link distance of 1-meter ($\gamma_{GU}$, $\gamma_{GB}$, and $\gamma_{UB}$)
REFERENCE_SNR_AT_1_METER = 1e4

''' Algorithmic model'''

# The maximum number of concurrent workers allowed in this evaluation
NUMBER_OF_WORKERS = 1024

# The convergence confidence level for optimization algorithms in this framework
BISECTION_CONVERGENCE_CONFIDENCE = 10

# The tolerance value for the bisection method to find the optimal value of $Z$ for rate adaptation
BISECTION_METHOD_TOLERANCE = 1e-10

''' Visualization model '''

# The Plotly API "markers-only" scatter plot mode
PLOTLY_MARKERS_MODE = 'markers'

# The Plotly API "lines and markers" scatter plot mode
PLOTLY_LINES_MARKERS_MODE = 'lines+markers'

''' Link specifications '''
# reference_coords, reference_height, node_height, link_name = [0.0, 0.0], UAV_HEIGHT, 0.0, 'GU'
reference_coords, reference_height, node_height, link_name = [0.0, 0.0], BASE_STATION_HEIGHT, 0.0, 'GB'
# reference_coords, reference_height, node_height, link_name = [0.0, 0.0], BASE_STATION_HEIGHT, UAV_HEIGHT, 'UB'

"""
Core operations
"""


def fz(z_):
    b = CHANNEL_BANDWIDTH
    return b * np.log2(1 + (0.5 * (z_ ** 2)))


def marcum_q(df, nc, x):
    return 1 - ncx2.cdf(x, df, nc)


def f(z_, *args):
    df, nc, y = args

    f_z = fz(z_)
    q_m = marcum_q(df, nc, (y * (z_ ** 2)))

    ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
    ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf

    return -ln_f_z - ln_q_m


def bisect(f_, df, nc, y, low, high, tolerance):
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


def z(gamma):
    b = CHANNEL_BANDWIDTH
    return np.sqrt(2 * ((2 ** (gamma / b)) - 1))


def u(gamma, d, los):
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    alpha, alpha_, kappa = LoS_PATH_LOSS_EXPONENT, NLoS_PATH_LOSS_EXPONENT, NLoS_ATTENUATION_CONSTANT
    return ((2 ** (gamma / b)) - 1) / (gamma_ * 1 if los else kappa * (d ** -alpha if los else -alpha_))


def evaluate_los_throughput(d, phi, r_star_los):
    k_1, k_2 = LoS_RICIAN_FACTOR_1, LoS_RICIAN_FACTOR_2,
    b, gamma_ = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER
    k, alpha = k_1 * np.exp(k_2 * phi), LoS_PATH_LOSS_EXPONENT
    df, nc, y, t = 2, (2 * k), (k + 1) * (1 / (gamma_ * (d ** -alpha))), BISECTION_METHOD_TOLERANCE

    z_star = bisect(f, df, nc, y, 0, z(b * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * gamma_ * (d ** -alpha))), t)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)


def evaluate_nlos_throughput(d, r_star_nlos):
    alpha_, t = NLoS_PATH_LOSS_EXPONENT, BISECTION_METHOD_TOLERANCE
    b, gamma_, kappa = CHANNEL_BANDWIDTH, REFERENCE_SNR_AT_1_METER, NLoS_ATTENUATION_CONSTANT

    df, nc, y = 2, 0, 1 / (gamma_ * (kappa * (d ** -alpha_)))

    z_star = bisect(f, df, nc, y, 0,
                    z(b * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * gamma_ * kappa * (d ** -alpha_))), t)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)


def calculate_adapted_throughput(d, phi, r_star_los, r_star_nlos):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.submit(evaluate_nlos_throughput, d, r_star_nlos)
        executor.submit(evaluate_los_throughput, d, phi, r_star_los)


# Run Trigger
if __name__ == '__main__':
    bs_height, uav_height = BASE_STATION_HEIGHT, UAV_HEIGHT
    z_1, z_2 = PROPAGATION_ENVIRONMENT_PARAMETER_1, PROPAGATION_ENVIRONMENT_PARAMETER_2
    cell_radius, num_levels, num_workers = CELL_RADIUS, NUMBER_OF_RADIUS_LEVELS, NUMBER_OF_WORKERS

    levels = np.linspace(start=0.0, stop=cell_radius, num=num_levels)
    angle_choice = np.random.choice(np.linspace(start=0.0, stop=2 * np.pi, num=10 * num_levels))
    nodes = [_ * np.squeeze(np.einsum('ji', np.vstack([np.cos(angle_choice), np.sin(angle_choice)]))) for _ in levels]

    xy_distances = tf.norm(tf.subtract(nodes, reference_coords), axis=1)
    relative_heights = tf.constant(abs(reference_height - node_height), shape=xy_distances.shape, dtype=tf.float64)

    distances = tf.sqrt(tf.add(tf.square(xy_distances), tf.square(relative_heights))).numpy()
    angles = tf.asin(tf.divide(relative_heights, distances)).numpy()

    los_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)
    nlos_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)

    phi_degrees = (180.0 / np.pi) * angles
    r_los, r_nlos = los_throughputs, nlos_throughputs
    p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
    p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)

    with ThreadPoolExecutor(max_workers=num_workers) as exeggutor:
        for index, (distance, angle) in enumerate(zip(distances, angles)):
            exeggutor.submit(calculate_adapted_throughput, distance, angle, r_los[index], r_nlos[index])

    r_avg = tf.add(tf.multiply(p_los, r_los), tf.multiply(p_nlos, r_nlos))

    x_vals, y_avg = xy_distances, r_avg.numpy() / 1e6

    delay_1Mb = tf.constant(1e6, dtype=tf.float64) / r_avg
    delay_10Mb = tf.constant(10e6, dtype=tf.float64) / r_avg
    delay_100Mb = tf.constant(100e6, dtype=tf.float64) / r_avg

    plot_trace_avg = graph_objs.Scatter(x=x_vals, y=y_avg, mode=PLOTLY_LINES_MARKERS_MODE)
    plot_layout_avg = dict(title=f'{link_name} Link | Rate Adapted Average Throughput in Mbps',
                           xaxis=dict(title='Tx-Rx Distance in m (projected at ground level)', autorange=True),
                           yaxis=dict(title=f'{link_name} Average Throughput in Mbps', type='log', autorange=True))

    fig_avg = dict(data=[plot_trace_avg], layout=plot_layout_avg)
    fig_avg_url = plotly.plotly.plot(fig_avg, filename=''.join([link_name, '_Average_Throughputs']), auto_open=False)
    print(f'[INFO] ChannelModel main: Average throughput plot for {link_name} links is given here - {fig_avg_url}.')

    plot_trace_delay_1Mb = graph_objs.Scatter(x=x_vals, y=delay_1Mb, name='1 Mb', mode=PLOTLY_LINES_MARKERS_MODE)
    plot_trace_delay_10Mb = graph_objs.Scatter(x=x_vals, y=delay_10Mb, name='10 Mb', mode=PLOTLY_LINES_MARKERS_MODE)
    plot_trace_delay_100Mb = graph_objs.Scatter(x=x_vals, y=delay_100Mb, name='100 Mb', mode=PLOTLY_LINES_MARKERS_MODE)

    plot_layout_delay = dict(title=f'{link_name} | Average Delays',
                             yaxis=dict(title=f'{link_name} Average Delay in seconds', autorange=True),
                             xaxis=dict(title='Tx-Rx Distance in m (projected at ground level)', autorange=True))

    fig_dly = dict(data=[plot_trace_delay_1Mb, plot_trace_delay_10Mb, plot_trace_delay_100Mb], layout=plot_layout_delay)
    fig_dly_url = plotly.plotly.plot(fig_dly, filename=''.join([link_name, '_Average_Delays']), auto_open=False)

    print(f'[INFO] ChannelModel main: Average Delay plot for {link_name} links is given here - {fig_dly_url}.')
