"""
This script evaluates the convergence performance of our novel Hierarchical Competitive Swarm Optimization (HCSO)
algorithm for UAV trajectory design -- employing the same deployment, communication, channel, and mobility power models
as those used in the proposed MAESTRO framework. A detailed description of HCSO can be found in our journal paper.

Reference Paper:
                @ARTICLE{HCSO,
                author={Keshavamurthy, Bharath and Bliss, Matthew and Michelusi, Nicolò},
                title={{MAESTRO-X: Distributed Orchestration of Rotary-Wing UAV Relay Swarms}},
                journal={{IEEE Transactions on Cognitive Communications and Networking}},
                month={6},
                year={2022},
                note={{Submitted}}

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import os

"""
Configuration-I: Tensorflow Logging
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit ' \
                             '/home/bkeshav1/workspace/repos/Odin/src/uav-mobility/soa-evaluations/HCSOConvergence.py'

import time
import json
import traceback
import numpy as np
import tensorflow as tf
from collections import namedtuple
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Deployment Parameters
"""

pi = np.pi
np.random.seed(6)
omega, zeta, epsilon = 1.0, 1.0, 1.0
h_bs, h_uav, h_gns = 80.0, 200.0, 0.0
a, m, m_ip, m_max, n = 1e3, 6, 2, 256, 400
r_bounds, th_bounds = (-a, a), (0, 2 * pi)
x_g = tf.constant([[-570.0, 601.0]], dtype=tf.float64)
bw, snr_0, a_los, a_nlos, kappa = 5e6, 1e4, 2.0, 2.8, 0.2
p_avg = np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[1]
k_1, k_2, z_1, z_2, conf, tol, m_post = 1.0, np.log(100) / 90.0, 9.61, 0.16, 10, 1e-5, m_ip * (m + 2)
utip, v0, p1, p2, p3, k_max, v_min, v_max, v_num = 200.0, 7.2, 580.65, 790.6715, 0.0073, 100, 0.0, 55.0, 25
nu, data_len, arr_rates = 0.99 / p_avg, [1e6, 10e6, 100e6][1], {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}
x_0, x_m = tf.constant([[400.0, -300.0]], dtype=tf.float64), tf.constant([[-387.50, 391.50]], dtype=tf.float64)

"""
UAV Mobility Power Computation Routine
"""


def mobility_pwr(v):
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
           (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


"""
Random Trajectories Generation 
"""


# noinspection PyMethodMayBeStatic
class RandomTrajectoriesGeneration(object):

    def __enter__(self):
        return self

    def __generate(self, traj):
        (r_min, r_max), (th_min, th_max) = r_bounds, th_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)
        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)), tf.multiply(r, tf.math.sin(theta))],
                                            axis=1), validate_shape=True, use_locking=True)

    def generate(self, n_w):
        trajs = tf.Variable(tf.zeros(shape=(int(n / 2), m, 2), dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for i in range(int(n / 2)):
                executor.submit(self.__generate, trajs[i, :])
        return trajs

    def __optimize(self, traj, opt_traj):
        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), (m_ip * len(i_s)), dtype=np.float64)
        tf.compat.v1.assign(opt_traj, tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
                            validate_shape=True, use_locking=True)

    def optimize(self, trajs, n_w):
        opt_trajs = tf.Variable(tf.zeros(shape=[int(n / 2), m_post, 2], dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for i in range(int(n / 2)):
                executor.submit(self.__optimize, trajs[i, :], opt_trajs[i, :])
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Exception Type = {exc_type} | Exception Value = {exc_val} | '
                  f'Traceback = {traceback.print_tb(exc_tb)}')


"""
Deterministic Trajectories Generation
"""


# noinspection PyMethodMayBeStatic
class DeterministicTrajectoriesGeneration(object):

    def __enter__(self):
        return self

    def generate(self):
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
                  f'Exception Type = {exc_type} | Exception Value = {exc_val} | '
                  f'Traceback = {traceback.print_tb(exc_tb)}')


"""
Link Performance Evaluator
"""


# noinspection PyMethodMayBeStatic
class LinkPerformanceEvaluator(object):

    def __init__(self):
        print('[INFO] LinkPerformanceEvaluator Initialization: Bringing things up...')
        # Nothing to do here...

    def __enter__(self):
        return self

    def __f_z(self, z):
        return bw * np.log2(1 + (0.5 * (z ** 2)))

    def __marcum_q(self, df, nc, x):
        return 1 - ncx2.cdf(x, df, nc)

    def __f(self, z, *args):
        df, nc, y = args
        f_z = self.__f_z(z)
        q_m = self.__marcum_q(df, nc, (y * (z ** 2)))
        ln_f_z = np.log(f_z) if f_z != 0.0 else -np.inf
        ln_q_m = np.log(q_m) if q_m != 0.0 else -np.inf
        return -ln_f_z - ln_q_m

    def __bisect(self, f, df, nc, y, lo, hi):
        args = (df, nc, y)
        mid, conv, c = 0.0, False, 0
        while not conv or c < conf:
            mid = (lo + hi) / 2
            if (f(lo, *args) * f(hi, *args)) > 0.0:
                lo = mid
            else:
                hi = mid
            conv = abs(lo - hi) < tol
            c += 1 if conv else -c
        return mid

    def __z(self, gamma):
        return np.sqrt(2 * ((2 ** (gamma / bw)) - 1))

    def __u(self, gamma, d, los):
        return ((2 ** (gamma / bw)) - 1) / (snr_0 * (lambda: kappa, lambda: 1)[los]() *
                                            (d ** (lambda: -a_nlos, lambda: -a_los)[los]()))

    def __los_throughput(self, d, phi, r_los):
        k = k_1 * np.exp(k_2 * phi)
        df, nc, y = 2, (2 * k), (k + 1) * (1 / (snr_0 * (d ** -a_los)))
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(bw * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) *
                                                                             snr_0 * (d ** -a_los))))
        tf.compat.v1.assign(r_los, self.__f_z(z_star), use_locking=True)

    def __nlos_throughput(self, d, r_nlos):
        df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -a_nlos)))
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(bw * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) *
                                                                             snr_0 * kappa * (d ** -a_nlos))))
        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), use_locking=True)

    def adapted_throughput(self, d, phi, r_los, r_nlos, n_w):
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__los_throughput, d, phi, r_los)
            executor.submit(self.__nlos_throughput, d, r_nlos)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformanceEvaluator Termination: Tearing things down - Exception Type = {exc_type} | '
              f'Exception Value = {exc_val} | Traceback = {exc_tb}')


"""
Core HCSO Routines
"""

min_pwr = mobility_pwr(22.0)
lpf = LinkPerformanceEvaluator()
penalty_vars = namedtuple('penalties_capsule', ['t_p_1', 't_p_2', 'e_p_1', 'e_p_2'])


def ip_wps(p_indices, p):
    m_ = len(p_indices)
    spl_x, spl_y = UnivariateSpline(p_indices, p[:, 0], s=0), UnivariateSpline(p_indices, p[:, 1], s=0)
    return tf.constant(list(zip(spl_x(np.linspace(0, m_ - 1, m_ip * m_, dtype=np.float64)),
                                spl_y(np.linspace(0, m_ - 1, m_ip * m_, dtype=np.float64)))))


def ip_vels(v_indices, v):
    m_, spl_v = len(v_indices), UnivariateSpline(v_indices, v, s=0)
    return spl_v(np.linspace(0, m_ - 1, m_ip * m_, dtype=np.float64))


def penalties(p_, v_, n_w):
    p = ip_wps([_ for _ in range(p_.shape[0])], p_)
    midpoint = int(p.shape[0] / 2)
    v = ip_vels([_ for _ in range(v_.shape[0])], v_)
    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1),
                  tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
    h_u = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)
    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
    phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()
    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu, phi_gu)):
            executor.submit(lpf.adapted_throughput, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i])
    phi_degrees_gu = (180.0 / pi) * phi_gu
    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))
    h_1 = data_len - tf.reduce_sum(tf.multiply(t[:midpoint], r_bar_gu))
    t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
    e_p_1 = (lambda: 0.0, lambda: (min_pwr * t_p_1))[h_1.numpy() > 0.0]()

    r_ub = tf.norm(p[midpoint:], axis=1)
    h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)
    d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
    phi_ub = tf.asin(tf.divide(h_ub, d_ub))
    r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_ub, _phi_ub) in enumerate(zip(d_ub, phi_ub)):
            executor.submit(lpf.adapted_throughput, _d_ub, _phi_ub, r_los_ub[_i], r_nlos_ub[_i])
    phi_degrees_ub = (180.0 / pi) * phi_ub
    p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
    p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
    r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))
    h_2 = data_len - tf.reduce_sum(tf.multiply(t[midpoint:], r_bar_ub[:-1]))
    t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
    e_p_2 = (lambda: 0.0, lambda: (min_pwr * t_p_2))[h_2.numpy() > 0.0]()
    return penalty_vars(t_p_1=t_p_1, t_p_2=t_p_2, e_p_1=e_p_1, e_p_2=e_p_2)


@tf.function
@tf.autograph.experimental.do_not_convert
def pwr_cost(v, n_w): return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


def comm_cost(p, v, f_hat, n_w):
    t_p_1, t_p_2, e_p_1, e_p_2 = penalties(p, v, n_w)
    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1),
                  tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))
    t__ = tf.reduce_sum(t) + t_p_1 + t_p_2
    e__ = tf.reduce_sum(tf.multiply(t, pwr_cost(v[:-1], n_w))) + e_p_1 + e_p_2
    tf.compat.v1.assign(f_hat, ((1.0 - nu * p_avg) * t__ + nu * e__), validate_shape=True, use_locking=True)


def update(p, v, u, w, t_j, t_j_1, p_bar, v_bar, f_hats, n_w):
    p_t_j, p_t_j_1, v_t_j, v_t_j_1 = p[t_j], p[t_j_1], v[t_j], v[t_j_1]
    u_t_j, u_t_j_1, w_t_j, w_t_j_1 = u[t_j], u[t_j_1], w[t_j], w[t_j_1]
    f_hat_t_j, f_hat_t_j_1 = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=n_w) as executor:
        executor.submit(comm_cost, p_t_j, v_t_j, f_hat_t_j, n_w)
        executor.submit(comm_cost, p_t_j_1, v_t_j_1, f_hat_t_j_1, n_w)
    argmin__ = np.argmin([f_hat_t_j, f_hat_t_j_1])
    j_win, p_w, v_w, j_los, p_l, v_l, u_l, w_l = (t_j, p_t_j, v_t_j, t_j_1, p_t_j_1, v_t_j_1, u_t_j_1, w_t_j_1) \
        if (argmin__ == 0) else (t_j_1, p_t_j_1, v_t_j_1, t_j, p_t_j, v_t_j, u_t_j, w_t_j)
    r_j = tf.random.uniform(shape=[3, ], dtype=tf.float64)

    tf.compat.v1.assign(f_hats[t_j], f_hat_t_j, validate_shape=True, use_locking=True)
    u_j_los = tf.add_n([tf.multiply(r_j[0], u_l), tf.multiply(r_j[1], tf.subtract(p_w, p_l)),
                        omega * tf.multiply(r_j[2], tf.subtract(p_bar, p_l))])
    p_j_los = tf.add(p_l, u_j_los)
    tf.compat.v1.assign(p[j_los], p_j_los, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(u[j_los], u_j_los, validate_shape=True, use_locking=True)

    tf.compat.v1.assign(f_hats[t_j_1], f_hat_t_j_1, validate_shape=True, use_locking=True)
    w_j_los = tf.add_n([tf.multiply(r_j[0], w_l), tf.multiply(r_j[1], tf.subtract(v_w, v_l)),
                        omega * tf.multiply(r_j[2], tf.subtract(v_bar, v_l))])
    v_j_los = tf.clip_by_value(tf.add(v_l, w_j_los), v_min, v_max)
    tf.compat.v1.assign(v[j_los], v_j_los, validate_shape=True, use_locking=True)
    tf.compat.v1.assign(w[j_los], w_j_los, validate_shape=True, use_locking=True)


def cso(p, v, u, w, n_w):
    indices = [_ for _ in range(n)]
    k, p_bar, v_bar = 0, tf.reduce_mean(p, axis=0), tf.reduce_mean(v, axis=0)
    f_hats = tf.Variable(tf.zeros(shape=[n, ], dtype=tf.float64), dtype=tf.float64)

    while k <= k_max:
        t = tf.random.shuffle(indices)
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for j in range(0, n, 2):
                executor.submit(update, p, v, u, w, t[j], t[j + 1], p_bar, v_bar, f_hats, n_w)
        k += 1

    i = min(indices, key=lambda _k: f_hats[_k].numpy())
    return p[i], u[i], v[i], w[i], f_hats[i]


def hcso(p, v, u, w, n_w):
    n_, m_post_ = n, m_post
    i, p_star, u_star, v_star, w_star, f_star = 0, None, None, None, None, None

    while m_post_ < m_max * m_ip:
        n_, i = n_ - 20 * (i + 1), i + 1
        indices = [_ for _ in range(m_post_)]
        p_star, u_star, v_star, w_star, f_star = cso(p, v, u, w, n_w)
        p_tilde = tf.tile(tf.expand_dims(ip_wps(indices, p_star), axis=0), multiples=[n_, 1, 1])

        p, p_ = tf.Variable(tf.zeros(shape=p_tilde.shape, dtype=tf.float64), dtype=tf.float64), p_tilde[:, :-1, :]
        p_1, p_2 = tf.roll(p_tilde, shift=-1, axis=1)[:, :-1, :], tf.roll(p_tilde, shift=1, axis=1)[:, 1:, :]
        u = tf.Variable(tf.zeros(shape=p_tilde.shape, dtype=tf.float64), dtype=tf.float64)
        shape_p = [n_, (m_ip * m_post_) - 1, 2]

        scl_f = tf.expand_dims(tf.add(tf.square(tf.norm(p_tilde[:, -1, :], axis=1)), tf.multiply(
            zeta, tf.square(tf.norm(tf.subtract(p_tilde[:, -2, :], p_tilde[:, -1, :]), axis=1)))), axis=1)

        scales = tf.tile(tf.expand_dims(tf.add(tf.square(tf.norm(tf.subtract(
            p_1, p_), axis=2)), tf.multiply(zeta, tf.square(tf.norm(tf.subtract(p_2, p_),
                                                                    axis=2)))), axis=2), multiples=[1, 1, 2])

        p_tilde_r = tf.random.normal(shape=shape_p, stddev=tf.sqrt(scales),
                                     mean=tf.zeros(shape=shape_p, dtype=tf.float64), dtype=tf.float64)
        tf.compat.v1.assign(p[:, :-1, :], tf.add(p_tilde[:, :-1, :], p_tilde_r), validate_shape=True, use_locking=True)
        tf.compat.v1.assign(p[:, -1, :], tf.add(p_tilde[:, -1, :], tf.random.normal(
            shape=[scl_f.shape[0], 2], mean=tf.zeros(shape=[scl_f.shape[0], 2], dtype=tf.float64),
            stddev=tf.tile(tf.sqrt(scl_f), multiples=[1, 2]), dtype=tf.float64)), validate_shape=True, use_locking=True)

        tf.compat.v1.assign(u, tf.tile(tf.expand_dims(ip_wps(indices, u_star), axis=0),
                                       multiples=[n_, 1, 1]), validate_shape=True, use_locking=True)

        v_tilde = tf.tile(tf.expand_dims(ip_vels(indices, v_star), axis=0), multiples=[n_, 1])
        v = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
        w = tf.Variable(tf.zeros(shape=v_tilde.shape, dtype=tf.float64), dtype=tf.float64)
        shape_v = [n_, m_ip * m_post_]

        v_tilde_r = tf.random.normal(shape=shape_v,
                                     mean=tf.zeros(shape=shape_v, dtype=tf.float64),
                                     stddev=tf.sqrt(tf.multiply(epsilon * ((v_max - v_min) ** 2),
                                                    tf.ones(shape=shape_v, dtype=tf.float64))), dtype=tf.float64)

        tf.compat.v1.assign(v, tf.clip_by_value(tf.add(
            v_tilde, v_tilde_r), v_min, v_max), validate_shape=True, use_locking=True)
        tf.compat.v1.assign(w, tf.tile(tf.expand_dims(ip_vels(indices, w_star), axis=0),
                                       multiples=[n_, 1]), validate_shape=False, use_locking=True)
        m_post_ *= m_ip
    return p_star, u_star, v_star, w_star, f_star


def analyze(n_w):
    vals = np.linspace(v_min, v_max, v_num)
    c, lagrs, f_star_old, f_star_new = 0, {}, 0.0, 0.0
    v = tf.Variable(np.random.choice(vals, size=[n, m_post]))
    w = tf.Variable(np.random.choice(vals, size=[n, m_post]))
    u = tf.Variable(np.random.choice(vals, size=[n, m_post, 2]))

    def converged():
        return f_star_new.numpy() - f_star_old.numpy() > tol

    try:
        r_gen = RandomTrajectoriesGeneration()
        d_gen = DeterministicTrajectoriesGeneration()
        p = tf.concat([d_gen.generate(), r_gen.optimize(r_gen.generate(n_w), n_w)], axis=0)

        while c < conf or not converged():
            f_star_old = f_star_new
            # noinspection PyTypeChecker
            p_star, u_star, v_star, w_star, f_star_new = hcso(p, v, u, w, n_w)
            lagrs[time.monotonic()] = f_star_new

        with open('../perf-logs/hcso-convergence.log', 'w') as f:
            json.dump(lagrs, f)
        return True
    except Exception as e:
        print('[ERROR] HCSOConvergence analyze: Exception caught while analyzing the convergence properties of the '
              'Hierarchical Competitive Swarm Optimization algorithm - {}'.format(traceback.print_tb(e.__traceback__)))
        return False


# Run Trigger
if __name__ == '__main__':
    print(f'[INFO] HCSOConvergence main: Hierarchical Competitive Swarm Optimization | Data Length = {data_len} Mb | '
          f'UAV Average Power Constraint = {p_avg} Watts | Convergence Analysis Status = {analyze(256)}')
