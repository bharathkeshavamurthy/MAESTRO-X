"""
This script encapsulates the operations involved in visualizing the polar plots of an HCSO-optimized trajectory for a
specific comm state-action pair, along with delay-power curves for different values of the HCSO 'alpha' parameter.

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

import re
import plotly
import warnings
import traceback
import numpy as np
import tensorflow as tf
from scipy import interpolate
import plotly.graph_objs as graph_objs
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# Numpy random seed
np.random.seed(6)

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Configurations-II: Simulation parameters 
"""

pi = np.pi
bw, n_c = 40e6, 8
ip_dir = '../../../logs/policies/'
a, n_w, m, m_ip = 1e3, 1024, 32, 2
bw_, ap, ap_, kp = bw / n_c, 2.0, 2.8, 0.2
k_1, k_2, z_1, z_2 = 1.0, np.log(100) / 90.0, 9.61, 0.16
sg_wsize1, sg_wsize2, sg_poly_order1, sg_poly_order2 = 5, 5, 3, 3
hcso_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)
utip, v0, p1, p2, p3, v_min, v_max = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0
plotly.tools.set_credentials_file(username='bkeshav1', api_key='BCvYNi3LNNXgfpDGEpo0')
snr_0, ra_tol, ra_conf, h_bs, h_u, h_gn = linear((5e6 * 40) / bw_), 1e-10, 10, 80.0, 200.0, 0.0
data_len, p_avg = [1e6, 10e6, 100e6][1], np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[1]

"""
Configurations-III: Deployment settings
"""

''' BS deployment '''
x_bs = tf.constant([0.0, 0.0], dtype=tf.float64)

"""
Utilities
"""


# noinspection PyMethodMayBeStatic
class DeterministicTrajectoriesGeneration(object):

    def __init__(self, x_0, x_m, m_post):
        self.x_0 = x_0
        self.x_m = x_m
        self.m_post = m_post

    def __enter__(self):
        return self

    def generate(self):
        x_0, x_m, m_post = self.x_0, self.x_m, self.m_post

        x_mid = tf.divide(tf.add(x_0, x_m), 2)
        x_mid_0 = tf.divide(tf.add(x_0, x_mid), 2)
        x_mid_m = tf.divide(tf.add(x_mid, x_m), 2)
        traj = tf.concat([x_mid_0, x_mid, x_mid_m], axis=0)

        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), m_post, dtype=np.float64)

        return tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + (p3 * (v ** 3)) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5)


min_pwr = mobility_pwr(22.0)
max_pwr = mobility_pwr(55.0)


class LinkPerformance(object):
    """
    Link performance
    """

    @staticmethod
    def f_z(z):
        return bw_ * np.log2(1.0 + (0.5 * (z ** 2.0)))

    @staticmethod
    def q(df, nc, x):
        return 1.0 - ncx2.cdf(x, df, nc)

    def f(self, z, *args_):
        df, nc, y = args_
        f_z = self.f_z(z)
        q_m = self.q(df, nc, (y * (z ** 2.0)))
        ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
        ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf

        return -ln_f_z - ln_q_m

    @staticmethod
    def bisect(f, df, nc, y, low, high):
        args_ = (df, nc, y)
        mid, conv, conf = 0.0, False, 0

        while not conv or conf < ra_conf:
            mid = (high + low) / 2.0
            if f(low, *args_) * f(high, *args_) > 0.0:
                low = mid
            else:
                high = mid
            conv = abs(high - low) < ra_tol
            conf += 1 if conv else -conf

        return mid

    @staticmethod
    def z(gm):
        return np.sqrt(2.0 * ((2.0 ** (gm / bw_)) - 1.0))

    @staticmethod
    def u(gm, d, los):
        return ((2.0 ** (gm / bw_)) - 1.0) / (snr_0 * 1.0 if los else kp * (d ** -ap if los else -ap_))

    def los_tgpt(self, d, phi, r_los):
        k = k_1 * np.exp(k_2 * phi)
        ppf_pt = rice.ppf(0.9999999999, k) ** 2.0
        df, nc, y = 2, (2 * k), (k + 1.0) * (1.0 / (snr_0 * (d ** -ap)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * snr_0 * (d ** -ap))))

        tf.compat.v1.assign(r_los, self.f_z(z_star), use_locking=True)

    def nlos_tgpt(self, d, r_nlos):
        ppf_pt = rayleigh.ppf(0.9999999999) ** 2.0
        df, nc, y = 2, 0, 1.0 / (snr_0 * (kp * (d ** -ap_)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * snr_0 * kp * (d ** -ap_))))

        tf.compat.v1.assign(r_nlos, self.f_z(z_star), use_locking=True)

    def ra_tgpt(self, d, phi, r_los, r_nlos, num_workers):
        with ThreadPoolExecutor(num_workers) as executor:
            executor.submit(self.nlos_tgpt, d, r_nlos)
            executor.submit(self.los_tgpt, d, phi, r_los)


@tf.function(experimental_relax_shapes=True)
@tf.autograph.experimental.do_not_convert
def pwr_cost(v):
    return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


def decode(x_gn, d_p, d_v, d_lp):
    link_perf = LinkPerformance()

    r_gu = tf.norm(tf.subtract(d_p, tf.tile(tf.expand_dims(x_gn, axis=0), multiples=[d_p.shape[0], 1])), axis=1)
    h_uav = tf.constant(h_u, shape=r_gu.shape, dtype=tf.float64)

    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_uav)))
    phi_gu = tf.asin(tf.divide(h_uav, d_gu))

    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu.numpy(), phi_gu.numpy())):
            executor.submit(link_perf.ra_tgpt, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i], n_w)

    phi_degrees_gu = (180.0 / pi) * phi_gu

    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)

    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

    t = tf.divide(tf.norm(tf.roll(d_p, shift=-1, axis=0)[:-1, :] - d_p[:-1, :], axis=1),
                  tf.where(tf.equal(d_v[:-1], 0.0), tf.ones_like(d_v[:-1]), d_v[:-1]))

    h_xtra = data_len - d_lp - tf.reduce_sum(tf.multiply(t, r_bar_gu[:-1]))

    t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
    e_p = (lambda: 0.0, lambda: (min_pwr * t_p))[h_xtra.numpy() > 0.0]()

    return tf.reduce_sum(t), t_p, tf.reduce_sum(tf.multiply(t, pwr_cost(d_v[:-1]))), e_p


def forward(f_p, f_v, f_lp):
    link_perf = LinkPerformance()

    r_ub = tf.norm(tf.subtract(f_p, tf.tile(tf.expand_dims(x_bs, axis=0), multiples=[f_p.shape[0], 1])), axis=1)
    h_ub = tf.constant(abs(h_u - h_bs), shape=r_ub.shape, dtype=tf.float64)

    d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
    phi_ub = tf.asin(tf.divide(h_ub, d_ub))

    r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_ub, _phi_ub) in enumerate(zip(d_ub, phi_ub)):
            executor.submit(link_perf.ra_tgpt, _d_ub, _phi_ub, r_los_ub[_i], r_nlos_ub[_i], n_w)

    phi_degrees_ub = (180.0 / pi) * phi_ub

    p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
    p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)

    r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))

    t = tf.divide(tf.norm(tf.roll(f_p, shift=-1, axis=0)[:-1, :] - f_p[:-1, :], axis=1),
                  tf.where(tf.equal(f_v[:-1], 0.0), tf.ones_like(f_v[:-1]), f_v[:-1]))

    h_xtra = data_len - f_lp - tf.reduce_sum(tf.multiply(t, r_bar_ub[:-1]))

    t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
    e_p = (lambda: 0.0, lambda: (min_pwr * t_p))[h_xtra.numpy() > 0.0]()

    return tf.reduce_sum(t), t_p, tf.reduce_sum(tf.multiply(t, pwr_cost(f_v[:-1]))), e_p


def service(x_gn, *args_):
    d_p, d_v, d_lp, f_p, f_v, f_lp = args_
    t_ub, t_p_ub, e_ub, e_p_ub = forward(f_p, f_v, f_lp)
    t_gu, t_p_gu, e_gu, e_p_gu = decode(x_gn, d_p, d_v, d_lp)
    return np.mean([t_gu, t_ub]), np.mean([e_gu, e_ub])


# Run Trigger
if __name__ == '__main__':
    read_trajs = []
    traj_files = [f'{ip_dir}{int(data_len / 1e6)}_new/{_h_a}/trajs/0.log' for _h_a in hcso_alphas]

    for traj_file in traj_files:
        with open(traj_file, 'r') as file:
            lines = file.readlines()
            # noinspection RegExpUnnecessaryNonCapturingGroup
            file_v_star = tf.strings.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)',
                                                          lines[2].strip()), tf.float64)
            # noinspection RegExpUnnecessaryNonCapturingGroup
            file_p_star_ = tf.strings.to_number(re.findall(r'-?\d*\.?\d+e[+-]?\d+|[-+]?(?:\d*\.*\d+)',
                                                           lines[3].strip().replace('\\n', '')), tf.float64)

        p_star_arr = []
        for _idx in range(0, file_p_star_.shape[0], 2):
            p_star_arr.append([file_p_star_[_idx], file_p_star_[_idx + 1]])

        file_p_star = tf.Variable(p_star_arr, dtype=tf.float64)

        read_trajs.append((file_p_star, file_v_star))

    trajs = []
    num_trajs = len(read_trajs)
    x_u = tf.constant([500.0, 0.0], dtype=tf.float64)
    x_g = tf.constant([-611.0, 130.0], dtype=tf.float64)
    x_f = tf.constant([-225.09, 124.05], dtype=tf.float64)

    f_hats = tf.Variable(tf.zeros(shape=[num_trajs, ], dtype=tf.float64), dtype=tf.float64)
    deltas = tf.Variable(tf.zeros(shape=[num_trajs, ], dtype=tf.float64), dtype=tf.float64)
    p_usages = tf.Variable(tf.zeros(shape=[num_trajs, ], dtype=tf.float64), dtype=tf.float64)

    for traj_idx in range(num_trajs):
        h_alpha = hcso_alphas[traj_idx]

        p_star_, v_star_ = read_trajs[traj_idx]

        trajs.append((p_star_, v_star_))

        midpoint = int(p_star_.shape[0] / 2)

        s_time, s_nrg = service(x_g, p_star_[:midpoint], v_star_[:midpoint],
                                0.0, p_star_[midpoint:], v_star_[midpoint:], 0.0)

        print(f'{h_alpha}: {s_time} s | {s_nrg / s_time} W')

        tf.compat.v1.assign(deltas[traj_idx], s_time, validate_shape=True, use_locking=True)

        tf.compat.v1.assign(f_hats[traj_idx], ((1.0 - (2.0 * h_alpha)) * s_time) +
                            ((h_alpha / max_pwr) * s_nrg), validate_shape=True, use_locking=True)

        tf.compat.v1.assign(p_usages[traj_idx], s_nrg / s_time, validate_shape=True, use_locking=True)

    p_star, v_star = trajs[tf.argmin(tf.abs(f_hats), axis=0)]

    fn_d = interpolate.interp1d(hcso_alphas, deltas.numpy())
    fn_p = interpolate.interp1d(hcso_alphas, p_usages.numpy())

    hcso_alphas_new = np.arange(start=0.0, stop=1.0, step=0.01)

    plot_layout = dict(title='HCSO Alpha Metric Cost Analysis',
                       yaxis=dict(title='Power Consumption in W', autorange=True),
                       xaxis=dict(title='Communication Service Delay in s', autorange=True))

    plot_data = graph_objs.Scatter(x=fn_d(hcso_alphas_new), y=fn_p(hcso_alphas_new), mode='markers')

    fig = dict(data=[plot_data], layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='HCSO_Alpha_Cost_Analysis', auto_open=False)
    print('[INFO] HCSOVisualizations main: The plot of the HCSO Cost versus HCSO Alpha analysis '
          f'for the UAV at {x_u.numpy()} and the GN at {x_g.numpy()} is available at - {fig_url}.')

    traj_plot_data, vel_plot_data = list(), list()

    vel_plot_data.append(graph_objs.Scatter(x=v_star.numpy(), y=v_star.numpy(), mode='markers'))

    traj_plot_data.append(graph_objs.Scatter(x=p_star[:, 0].numpy(), y=p_star[:, 1].numpy(),
                                             mode='markers', name='UAV Optimal Trajectory'))

    vel_plot_layout = dict(title='Optimal HCSO-determined UAV Velocities',
                           xaxis=dict(title='v (in m/s)'), yaxis=dict(title='v (in m/s)'))

    traj_plot_layout = dict(title='Optimal HCSO-determined UAV Trajectory',
                            xaxis=dict(title='x (in m)'), yaxis=dict(title='y (in m)'))

    fig_v = dict(data=vel_plot_data, layout=vel_plot_layout)
    fig_t = dict(data=traj_plot_data, layout=traj_plot_layout)
    fig_url_v = plotly.plotly.plot(fig_v, filename='HCSO_UAV_Velocity', auto_open=False)
    fig_url_t = plotly.plotly.plot(fig_t, filename='HCSO_UAV_Trajectory', auto_open=False)

    print('[INFO] HCSOVisualizations main: The plot of the optimal HCSO UAV trajectory '
          f'for the GN at {x_g.numpy()} [m, m], with Initial UAV Position = {x_u.numpy()} '
          f'[m, m] and Terminal UAV Position = {x_f.numpy()} [m, m], is available at - {fig_url_t}.')

    print('[INFO] HCSOVisualizations main: The plot of the optimal HCSO UAV velocities '
          f'for the GN at {x_g.numpy()} [m, m], with Initial UAV Position = {x_u.numpy()} '
          f'[m, m] and Terminal UAV Position = {x_f.numpy()} [m, m], is available at - {fig_url_v}.')
