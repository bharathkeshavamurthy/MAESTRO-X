"""
This script evaluates the convergence of the Competitive Swarm Optimization (CSO) algorithm as described in our paper.

Given the initial UAV position (x_U, y_U, H_U) and the "optimal" final UAV position to serve a GN request that arose,
the CSO algorithm determines the UAV's "optimal" trajectory segments (and their corresponding forward velocities).

This script evaluates the convergence performance of this "path planning" for benchmarking w.r.t PSO, SCA, and HCSO.

Reference Paper:

    @ARTICLE{CSO,
      author={Cheng, Ran and Jin, Yaochu},
      journal={IEEE Transactions on Cybernetics},
      title={A Competitive Swarm Optimizer for Large Scale Optimization},
      year={2015},
      volume={45},
      number={2},
      pages={191-204},
      doi={10.1109/TCYB.2014.2322602}}

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

import time
import json
import warnings
import traceback
import numpy as np
import tensorflow as tf
from collections import namedtuple
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Miscellaneous
"""

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-II: Simulation parameters
"""
np.random.seed(6)
bw, n_c = 20e6, 4
n_xu, n_xb = 3, 10
pi, bw_ = np.pi, bw / n_c
snr_0 = linear((5e6 * 40) / bw_)
a, m, m_ip, n = 1e3, 126, 2, 400
a_los, a_nlos, kappa = 2.0, 2.8, 0.2
h_bs, h_uav, h_gns = 80.0, 200.0, 0.0
r_bounds, th_bounds = (-a, a), (0, 2 * pi)
x_g = tf.constant([[-570.0, 601.0]], dtype=tf.float64)
output_log = '../../../logs/evaluations/cso_convergence.log'
p_avg, cso_conf, cso_tol = np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[1], 5, 1e-5
k_1, k_2, z_1, z_2, ra_conf, ra_tol, m_post = 1.0, np.log(100) / 90.0, 9.61, 0.16, 10, 1e-10, m_ip * (m + 2)
nu, data_len, arr_rates = 0.99 / p_avg, [1e6, 10e6, 100e6][1], {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}
x_0, x_m = tf.constant([[400.0, -300.0]], dtype=tf.float64), tf.constant([[-387.50, 391.50]], dtype=tf.float64)
utip, v0, p1, p2, p3, k_max, v_min, v_max, v_num, omega = 200.0, 7.2, 580.65, 790.6715, 0.0073, 100, 0.0, 55.0, 25, 1.0

"""
Utilities
"""


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


# noinspection PyMethodMayBeStatic
class RandomTrajectoriesGeneration(object):
    """
    Random trajectories generation
    """

    def __enter__(self):
        return self

    def __generate(self, traj):
        (r_min, r_max), (th_min, th_max) = r_bounds, th_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)

        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)),
                                             tf.multiply(r, tf.math.sin(theta))],
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
            print('[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


# noinspection PyMethodMayBeStatic
class DeterministicTrajectoriesGeneration(object):
    """
    Deterministic trajectories generation
    """

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
            print('[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


# noinspection PyMethodMayBeStatic
class LinkPerformance(object):
    """
    Link performance
    """

    def __enter__(self):
        return self

    def __f_z(self, z):
        return bw_ * np.log2(1 + (0.5 * (z ** 2)))

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

        while not conv or c < ra_conf:
            mid = (lo + hi) / 2
            if (f(lo, *args) * f(hi, *args)) > 0.0:
                lo = mid
            else:
                hi = mid
            conv = abs(lo - hi) < ra_tol
            c += 1 if conv else -c

        return mid

    def __z(self, gamma):
        return np.sqrt(2 * ((2 ** (gamma / bw_)) - 1))

    def __u(self, gamma, d, los):
        return ((2 ** (gamma / bw_)) - 1) / (snr_0 * 1 if los else kappa * (d ** -a_los if los else -a_nlos))

    def __los_throughput(self, d, phi, r_los):
        k = k_1 * np.exp(k_2 * phi)
        df, nc, y = 2, (2 * k), (k + 1) * (1 / (snr_0 * (d ** -a_los)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(bw_ * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * snr_0 * (d ** -a_los))))

        tf.compat.v1.assign(r_los, self.__f_z(z_star), use_locking=True)

    def __nlos_throughput(self, d, r_nlos):
        df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -a_nlos)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(bw_ * np.log2(1 + (
                                       rayleigh.ppf(0.9999999999) ** 2) * snr_0 * kappa * (d ** -a_nlos))))

        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), use_locking=True)

    def adapted_throughput(self, d, phi, r_los, r_nlos, n_w):
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__nlos_throughput, d, r_nlos)
            executor.submit(self.__los_throughput, d, phi, r_los)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] LinkPerformance Termination: Tearing things down - '
              f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {exc_tb}.')


"""
Core operations
"""

lperf = LinkPerformance()
min_pwr = mobility_pwr(22.0)
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
    v = ip_vels([_ for _ in range(v_.shape[0])], v_)
    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1),
                  tf.where(tf.equal(v[:-1], 0.0), tf.ones_like(v[:-1]), v[:-1]))

    midpoint = int(p.shape[0] / 2)
    r_gu = tf.norm(tf.subtract(p[:midpoint], x_g), axis=1)
    h_u = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)

    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_u))).numpy()
    phi_gu = tf.asin(tf.divide(h_u, d_gu)).numpy()

    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu, phi_gu)):
            executor.submit(lperf.adapted_throughput, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i])

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
            executor.submit(lperf.adapted_throughput, _d_ub, _phi_ub, r_los_ub[_i], r_nlos_ub[_i])

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
def pwr_cost(v, n_w):
    return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


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
        if argmin__ == 0 else (t_j_1, p_t_j_1, v_t_j_1, t_j, p_t_j, v_t_j, u_t_j, w_t_j)

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


def analyze(n_w):
    vals = np.linspace(v_min, v_max, v_num)
    v = tf.Variable(np.random.choice(vals, size=[n, m_post]))
    w = tf.Variable(np.random.choice(vals, size=[n, m_post]))
    u = tf.Variable(np.random.choice(vals, size=[n, m_post, 2]))
    c, conv, lagrs, f_star_old, f_star_new = 0, False, {}, 0.0, 0.0

    try:

        r_gen = RandomTrajectoriesGeneration()
        d_gen = DeterministicTrajectoriesGeneration()
        p = tf.concat([d_gen.generate(), r_gen.optimize(r_gen.generate(n_w), n_w)], axis=0)

        while not conv or c < cso_conf:
            f_star_old = f_star_new

            # noinspection PyTypeChecker
            p_star, u_star, v_star, w_star, f_star_new = cso(p, v, u, w, n_w)

            lagrs[time.monotonic()] = f_star_new

            conv = np.abs(f_star_new.numpy() - f_star_old.numpy()) < cso_tol
            c += 1 if conv else -c

        with open(output_log, 'w') as f:
            json.dump(lagrs, f)
        return True

    except Exception as e:
        print('[ERROR] CSOConvergence analyze: Exception caught while analyzing the convergence properties '
              f'of the Competitive Swarm Optimization (CSO) algorithm - {traceback.print_tb(e.__traceback__)}.')
        return False


# Run Trigger
if __name__ == '__main__':
    print(f'[INFO] CSOConvergence main: Competitive Swarm Optimization | Data Length = {data_len / 1e6} Mb | '
          f'UAV Average Power Constraint = {p_avg / 1e3} kW | CSO Convergence Analysis Status = {analyze(1024)}.')
