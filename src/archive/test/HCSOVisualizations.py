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
import numpy as np
import tensorflow as tf
import plotly.graph_objs as graph_objs
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Simulation parameters 
"""

pi = np.pi
np.random.seed(6)
bw, n_c = 40e6, 8
ip_dir = '../../../logs/policies/'
a, n_w, m, m_ip = 1e3, 1024, 32, 2
bw_, ap, ap_, kp = bw / n_c, 2.0, 2.8, 0.2
hcso_alphas = np.arange(start=0.0, stop=1.1, step=0.1)
k_1, k_2, z_1, z_2 = 1.0, np.log(100) / 90.0, 9.61, 0.16
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

''' Comm state-action pair under analysis '''
c_state, c_action = [500.0, 500.0, pi], 250.0

"""
Utilities
"""


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


def direct(x_gn):
    link_perf = LinkPerformance()

    d_gb = np.sqrt(np.add(np.square(h_bs), np.square(np.linalg.norm(x_gn.numpy()))))
    phi_gb = np.arcsin(np.divide(h_bs, d_gb))

    r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

    link_perf.ra_tgpt(d_gb, phi_gb, r_los_gb, r_nlos_gb, n_w)

    phi_degrees_gb = (180.0 / pi) * phi_gb

    p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
    p_nlos_gb = 1 - p_los_gb

    r_bar_gb = np.add(np.multiply(p_los_gb, r_los_gb.numpy()), np.multiply(p_nlos_gb, r_nlos_gb.numpy()))

    return data_len / r_bar_gb, 0.0


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
    return t_gu + t_p_gu + t_ub + t_p_ub, e_gu + e_p_gu + e_ub + e_p_ub


# Run Trigger
if __name__ == '__main__':
    read_trajs = []
    traj_files = [f'{ip_dir}{int(data_len / 1e6)}{_h_a}/trajs/0.log' for _h_a in hcso_alphas]

    for traj_file in traj_files:
        args = []

        with open(traj_file, 'r') as file:
            for line in file.readlines():
                # noinspection RegExpUnnecessaryNonCapturingGroup
                args.append(tf.string.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)', line.strip()), tf.float64))

        file_v_star, file_p_star = args[2:]
        read_trajs.append((file_p_star, file_v_star))

    r_u_ = c_action
    r_u, r_g, psi_gu = c_state
    num_trajs = len(read_trajs)
    x_u = tf.constant([r_u, 0.0], dtype=tf.float64)
    f_hats = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
    deltas = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
    e_usages = tf.Variable(tf.zeros(shape=[num_trajs, ]), dtype=tf.float64)
    x_g = tf.constant([r_g * np.cos(psi_gu), r_g * np.sin(psi_gu)], dtype=tf.float64)

    for traj_idx in range(num_trajs):
        h_alpha = hcso_alphas[traj_idx]
        with ThreadPoolExecutor(max_workers=n_w) as exxeggutor:
            p_star_, v_star_ = read_trajs[traj_idx]
            s_time, s_nrg = service(x_g)
            tf.compat.v1.assign(f_hats[traj_idx], ((1.0 - (2.0 * h_alpha)) * s_time) +
                                ((h_alpha / max_pwr) * s_nrg), validate_shape=True, use_locking=True)

    min_traj_idx = tf.argmin(f_hats, axis=0)
    p_star, v_star = read_trajs[min_traj_idx]

    x_m_1 = p_star[(m * m_ip) - 2, :]
    den = np.linalg.norm(x_m_1.numpy(), axis=1)

    if den == 0.0:
        x_m_updated = tf.Variable([r_u_, 0.0], dtype=tf.float64)
    else:
        x_m_updated = tf.multiply(r_u_, tf.divide(x_m_1, den))

    tf.compat.v1.assign(p_star[-1, :], x_m_updated)

    plot_layout = dict(title='HCSO Alpha Metric Cost Analysis',
                       xaxis=dict(title='Cost', autorange=True),
                       yaxis=dict(title='HCSO Alpha', autorange=True))
    plot_data = graph_objs.Scatter(x=hcso_alphas, y=f_hats.numpy(), mode='lines+markers')

    fig = dict(data=[plot_data], layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='HCSO_Alpha_Cost_Analysis', auto_open=False)
    print('[INFO] HCSOVisualizations main: The plot of the HCSO Cost versus HCSO Alpha '
          f'for the UAV at {r_u.numpy()} and GN at {x_g.numpy()} is available at - {fig_url}.')

    traj_plot_data = list()

    traj_plot_data.append(graph_objs.Scatter(x=[x_u[0].numpy()], y=[x_u[1].numpy()],
                                             mode='markers', name='UAV Initial Position ' + str(x_u.numpy())))

    traj_plot_data.append(graph_objs.Scatter(x=[x_g[0].numpy()], y=[x_g[1].numpy()],
                                             mode='markers', name='GN Position ' + str(x_g.numpy())))

    traj_plot_data.append(graph_objs.Scatter(x=p_star[:, 0].numpy(), y=p_star[:, 1].numpy(),
                                             mode='lines+markers', name='UAV Optimal Trajectory'))

    plot_layout = dict(title='Optimal HCSO-determined UAV Trajectory',
                       xaxis=dict(title='x (in m)'), yaxis=dict(title='y (in m)'))

    fig = dict(data=traj_plot_data, layout=plot_layout)
    fig_url = plotly.plotly.plot(fig, filename='HCSO_UAV_Trajectory', auto_open=False)
    print('[INFO] HCSOVisualizations main: The plot of the optimal HCSO-determined UAV trajectory '
          f'for the GN at {x_g.numpy()} [m, m], with Initial UAV Position = {x_u.numpy()} '
          f'[m, m] and Terminal UAV Position = {x_m_updated.numpy()} [m, m], is available at - {fig_url}.')
