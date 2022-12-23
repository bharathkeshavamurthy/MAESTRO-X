"""
This script encapsulates the operations involved in computing the lower bound on GN service latencies.

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
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# Numpy random seed
np.random.seed(6)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Simulation parameters
"""

n_w, depl_env, bw, radius, num_levels, bs_ht, gn_ht = 1024, 'rural', 20e6, 1e3, int(1e4), 80.0, 0.0
payload_size, alpha, alpha_, kappa, ra_tol, ra_conf = [1e6, 10e6, 100e6][0], 2.0, 2.8, 0.2, 1e-10, 10

'''
TODO: Change k_1, k_2, z_1, and z_2 according to the deployment environment
TODO: Change n_c according to the deployment environment (Verizon LTE/LTE-A/5G)
'''

if depl_env == 'rural':
    n_c, k_1, k_2, z_1, z_2 = 2, 1.0, np.log(100) / 90.0, 9.61, 0.16
elif depl_env == 'suburban':
    n_c, k_1, k_2, z_1, z_2 = 4, 1.0, np.log(100) / 90.0, 9.61, 0.16
else:
    n_c, k_1, k_2, z_1, z_2 = 10, 1.0, np.log(100) / 90.0, 9.61, 0.16

bw_ = bw / n_c
snr_0 = linear((5e6 * 40) / bw_)

# Assuming the UAV is directly on top of the GN (decode) and then immediately directly on top of the BS (forward)...
max_uav_decode_tgpt, max_uav_forward_tgpt = 19.7368e6, 26.8006e6

"""
Configurations-III: Deployment settings
"""

''' BS deployment '''
x_bs = [0.0, 0.0]

''' GNs deployment (discretization for integration) '''
levels = np.linspace(start=0.0, stop=radius, num=num_levels)
angle_choice = np.random.choice(np.linspace(start=0.0, stop=2 * np.pi, num=10 * num_levels))
gns = [_ * np.squeeze(np.einsum('ji', np.vstack([np.cos(angle_choice), np.sin(angle_choice)]))) for _ in levels]

"""
Core operations
"""


def fz(z_):
    return bw_ * np.log2(1 + (0.5 * (z_ ** 2)))


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
    mid, converged, conf, conf_th = 0.0, False, 0, ra_conf

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
    return np.sqrt(2 * ((2 ** (gamma / bw_)) - 1))


def u(gamma, d, los):
    return ((2 ** (gamma / bw_)) - 1) / (snr_0 * 1 if los else kappa * (d ** -alpha if los else -alpha_))


def evaluate_los_throughput(d, phi, r_star_los):
    k = k_1 * np.exp(k_2 * phi)
    df, nc, y = 2, (2 * k), (k + 1) * (1 / (snr_0 * (d ** -alpha)))

    z_star = bisect(f, df, nc, y, 0,
                    z(bw_ * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * snr_0 * (d ** -alpha))), ra_tol)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_los, gamma_star, validate_shape=True, use_locking=True)


def evaluate_nlos_throughput(d, r_star_nlos):
    df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -alpha_)))

    z_star = bisect(f, df, nc, y, 0,
                    z(bw_ * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) * snr_0 * kappa * (d ** -alpha_))), ra_tol)

    gamma_star = fz(z_star)
    tf.compat.v1.assign(r_star_nlos, gamma_star, validate_shape=True, use_locking=True)


def calculate_adapted_throughput(d, phi, r_star_los, r_star_nlos):
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        executor.submit(evaluate_nlos_throughput, d, r_star_nlos)
        executor.submit(evaluate_los_throughput, d, phi, r_star_los)


# Run Trigger
if __name__ == '__main__':
    delay = 0.0

    xy_distances = tf.norm(tf.subtract(gns, x_bs), axis=1)
    relative_heights = tf.constant(abs(bs_ht - gn_ht), shape=xy_distances.shape, dtype=tf.float64)

    distances = tf.sqrt(tf.add(tf.square(xy_distances), tf.square(relative_heights))).numpy()
    angles = tf.asin(tf.divide(relative_heights, distances)).numpy()

    los_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)
    nlos_throughputs = tf.Variable(tf.zeros(shape=distances.shape, dtype=tf.float64), dtype=tf.float64)

    phi_degrees = (180.0 / np.pi) * angles
    r_los, r_nlos = los_throughputs, nlos_throughputs
    p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
    p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)

    with ThreadPoolExecutor(max_workers=n_w) as exeggutor:
        for index, (distance, angle) in enumerate(zip(distances, angles)):
            exeggutor.submit(calculate_adapted_throughput, distance, angle, r_los[index], r_nlos[index])

    r_avg = tf.add(tf.multiply(p_los, r_los), tf.multiply(p_nlos, r_nlos))
    delays = tf.constant(payload_size, dtype=tf.float64) / r_avg

    for num in range(num_levels):
        delay += min(delays[num],
                     (payload_size / max_uav_decode_tgpt) +
                     (payload_size / max_uav_forward_tgpt)) * ((2 * levels[num]) / (radius ** 2))

    print(f'[INFO] LowerBound main: The lower bound on GN service latencies '
          f'for {payload_size / 1e6} Mb data payloads is computed to be {delay} seconds.')
