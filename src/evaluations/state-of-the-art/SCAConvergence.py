"""
This script evaluates the convergence of the Successive Convex Approximation (SCA) Algorithm as described in the state-
of-the-art paper referenced below -- to solve the optimization problem outlined in our paper.

Given the initial UAV position (x_U, y_U, H_U) and the "optimal" final UAV position to serve a GN request that arose,
the SCA algorithm determines the UAV's "optimal" trajectory segments (and their corresponding forward velocities). This
script then evaluates the convergence performance of this "path planning" for benchmarking w.r.t CSO, HCSO, and others.

Reference Paper:
                @ARTICLE{SCA,
                  author={Zeng, Yong and Xu, Jie and Zhang, Rui},
                  journal={IEEE Transactions on Wireless Communications},
                  title={Energy Minimization for Wireless Communication With Rotary-Wing UAV},
                  year={2019},
                  volume={18},
                  number={4},
                  pages={2329-2345},
                  doi={10.1109/TWC.2019.2902559}}

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

import time
import json
import warnings
import traceback
import numpy as np
import cvxpy as cp
import tensorflow as tf
from numpy.random import choice
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Configurations-II: Deployment Parameters
"""

pi = np.pi
np.random.seed(6)
a, m, m_ip, n = 1e3, 126, 2, 400
h_bs, h_uav, h_gns = 80.0, 200.0, 0.0
r_bounds, th_bounds = (-a, a), (0, 2 * pi)
x_g = tf.constant([[-570.0, 601.0]], dtype=tf.float64)
bw, snr_0, a_los, a_nlos, kappa = 5e6, 1e4, 2.0, 2.8, 0.2
k_1, k_2, z_1, z_2, conf, tol = 1.0, np.log(100) / 90.0, 9.61, 0.16, 10, 1e-5
p_avg, m_post = np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[1], (m + 2) * m_ip
utip, v0, p1, p2, p3, v_min, v_max, v_num, omega = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0, 25, 1.0
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
        tf.compat.v1.assign(r_los, self.__f_z(z_star), validate_shape=True, use_locking=True)

    def __nlos_throughput(self, d, r_nlos):
        df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -a_nlos)))
        z_star = self.__bisect(self.__f, df, nc, y, 0, self.__z(bw * np.log2(1 + (rayleigh.ppf(0.9999999999) ** 2) *
                                                                             snr_0 * kappa * (d ** -a_nlos))))
        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), validate_shape=True, use_locking=True)

    def adapted_throughput(self, d, phi, r_los, r_nlos, n_w):
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__nlos_throughput, d, r_nlos)
            executor.submit(self.__los_throughput, d, phi, r_los)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformanceEvaluator Termination: Tearing things down - Exception Type = {exc_type} | '
              f'Exception Value = {exc_val} | Traceback = {exc_tb}')


"""
Core SCA Routines
"""

hover_pwr = mobility_pwr(0.0)
lpf = LinkPerformanceEvaluator()


def setup(n_w):
    r_gen = RandomTrajectoriesGeneration()
    d_gen = DeterministicTrajectoriesGeneration()
    return tf.concat([d_gen.generate(), r_gen.optimize(r_gen.generate(n_w), n_w)], axis=0), \
        tf.Variable(choice(np.linspace(v_min, v_max, v_num), size=[m_post]), dtype=tf.float64)


# noinspection PyTypeChecker
def solve(p_, v_, n_w):
    p, v = setup(n_w)
    p_var = cp.Variable((m_post, 2), value=p.numpy())
    v_var = cp.Variable((m_post - 1,), value=tf.where(tf.equal(v[:-1], 0.0),
                                                      tf.ones_like(v[:-1]), v[:-1]).numpy())

    t_ = tf.divide(tf.norm(tf.roll(p_, shift=-1, axis=0)[:-1, :] -
                           p_[:-1, :], axis=1), tf.where(tf.equal(v_[:-1], 0.0), tf.ones_like(v_[:-1]), v_[:-1]))
    y_ = tf.subtract(tf.sqrt(1 + ((v_ ** 4) / (4 * v0 ** 4))), tf.divide(v_ ** 2, 2 * v0 ** 2))

    t = cp.Variable((m_post - 1,), value=np.divide(np.linalg.norm(p_var.value[1:, :] -
                                                                  p_var.value[:-1, :], axis=1), v_var.value))
    y = cp.Variable((m_post - 1,), value=np.sqrt(1 + ((v_var.value ** 4) /
                                                      (4 * v0 ** 4))) - ((v_var.value ** 2) / (2 * v0 ** 2)))

    midpoint_ = int(p_.shape[0] / 2)
    r_gu_ = tf.norm(tf.subtract(p_[:midpoint_], x_g), axis=1)
    h_gu_ = tf.constant(h_uav, shape=r_gu_.shape, dtype=tf.float64)
    d_gu_ = tf.sqrt(tf.add(tf.square(r_gu_), tf.square(h_gu_)))
    phi_gu_ = tf.asin(tf.divide(h_gu_, d_gu_))
    r_los_gu_ = tf.Variable(tf.zeros(shape=r_gu_.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu_ = tf.Variable(tf.zeros(shape=r_gu_.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu_, _phi_gu_) in enumerate(zip(d_gu_, phi_gu_)):
            executor.submit(lpf.adapted_throughput, _d_gu_, _phi_gu_, r_los_gu_[_i], r_nlos_gu_[_i], n_w)
    phi_degrees_gu_ = (180.0 / pi) * phi_gu_
    p_los_gu_ = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu_ - z_1))))
    p_nlos_gu_ = tf.subtract(tf.ones(shape=p_los_gu_.shape, dtype=tf.float64), p_los_gu_)
    r_bar_gu_ = tf.add(tf.multiply(p_los_gu_, r_los_gu_), tf.multiply(p_nlos_gu_, r_nlos_gu_))
    s_1_ = tf.reduce_sum(tf.multiply(t_[:midpoint_], r_bar_gu_))

    r_ub_ = tf.norm(p_[midpoint_:], axis=1)
    h_ub_ = tf.constant(abs(h_uav - h_bs), shape=r_ub_.shape, dtype=tf.float64)
    d_ub_ = tf.sqrt(tf.add(tf.square(r_ub_), tf.square(h_ub_)))
    phi_ub_ = tf.asin(tf.divide(h_ub_, d_ub_))
    r_los_ub_ = tf.Variable(tf.zeros(shape=r_ub_.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub_ = tf.Variable(tf.zeros(shape=r_ub_.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_ub_, _phi_ub_) in enumerate(zip(d_ub_, phi_ub_)):
            executor.submit(lpf.adapted_throughput, _d_ub_, _phi_ub_, r_los_ub_[_i], r_nlos_ub_[_i], n_w)
    phi_degrees_ub_ = (180.0 / pi) * phi_ub_
    p_los_ub_ = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub_ - z_1))))
    p_nlos_ub_ = tf.subtract(tf.ones(shape=p_los_ub_.shape, dtype=tf.float64), p_los_ub_)
    r_bar_ub_ = tf.add(tf.multiply(p_los_ub_, r_los_ub_), tf.multiply(p_nlos_ub_, r_nlos_ub_))
    s_2_ = tf.reduce_sum(tf.multiply(t_[midpoint_:], r_bar_ub_[:-1]))

    s_ = s_1_ + s_2_

    midpoint = int(p_var.value.shape[0] / 2)
    r_gu = tf.norm(tf.subtract(p_var.value[:midpoint], x_g), axis=1)
    h_gu = tf.constant(h_uav, shape=r_gu.shape, dtype=tf.float64)
    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_gu)))
    phi_gu = tf.asin(tf.divide(h_gu, d_gu))
    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu, phi_gu)):
            executor.submit(lpf.adapted_throughput, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i], n_w)
    phi_degrees_gu = (180.0 / pi) * phi_gu
    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))
    s_1 = tf.reduce_sum(tf.multiply(t.value[:midpoint], r_bar_gu))
    h_1 = data_len - s_1
    t_p_1 = (lambda: 0.0, lambda: (h_1 / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_1.numpy() > 0.0]()
    e_p_1 = (lambda: 0.0, lambda: (hover_pwr * t_p_1))[h_1.numpy() > 0.0]()

    r_ub = tf.norm(p_var.value[midpoint:], axis=1)
    h_ub = tf.constant(abs(h_uav - h_bs), shape=r_ub.shape, dtype=tf.float64)
    d_ub = tf.sqrt(tf.add(tf.square(r_ub), tf.square(h_ub)))
    phi_ub = tf.asin(tf.divide(h_ub, d_ub))
    r_los_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_ub = tf.Variable(tf.zeros(shape=r_ub.shape, dtype=tf.float64), dtype=tf.float64)
    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_ub, _phi_ub) in enumerate(zip(d_ub, phi_ub)):
            executor.submit(lpf.adapted_throughput, _d_ub, _phi_ub, r_los_ub[_i], r_nlos_ub[_i], n_w)
    phi_degrees_ub = (180.0 / pi) * phi_ub
    p_los_ub = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_ub - z_1))))
    p_nlos_ub = tf.subtract(tf.ones(shape=p_los_ub.shape, dtype=tf.float64), p_los_ub)
    r_bar_ub = tf.add(tf.multiply(p_los_ub, r_los_ub), tf.multiply(p_nlos_ub, r_nlos_ub))
    s_2 = tf.reduce_sum(tf.multiply(t.value[midpoint:], r_bar_ub[:-1]))
    h_2 = data_len - s_2
    t_p_2 = (lambda: 0.0, lambda: (h_2 / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_2.numpy() > 0.0]()
    e_p_2 = (lambda: 0.0, lambda: (hover_pwr * t_p_2))[h_2.numpy() > 0.0]()

    s = cp.Variable((1,), value=s_1 + s_2)

    d_ct = cp.sum(s_ + cp.multiply(np.sqrt(s_) * 2.0, cp.sqrt(s) - np.sqrt(s_)), axis=0)
    s_ct = cp.sum(y_ + cp.multiply(np.sqrt(y_) * 2.0, cp.sqrt(y) - np.sqrt(y_)) + v_ / (v0 ** 2) +
                  cp.multiply(2 * v_ / (v0 ** 2), v - v_), axis=0)

    pwr_cost = cp.transpose(p3 * cp.power(v_var, 3) + p2 * cp.sqrt(y) + p1 * (1 + (3 * cp.power(v_var,
                                                                                                2)) / (utip ** 2)))
    obj = ((1 - nu * p_avg) * (cp.sum(t, axis=0) + t_p_1 + t_p_2)) + (cp.multiply(nu, t @ pwr_cost) + e_p_1 + e_p_2)

    cts = [y >= 0, cp.power(y, -2) <= s_ct, data_len <= d_ct,
           p_var[0] == x_0, p_var[-1] == x_m, v_var >= v_min, v_var <= v_max]

    prb = cp.Problem(cp.Minimize(obj), cts)
    prb.solve('SCS', max_iters=int(1e6), eps_abs=1e-6, eps_rel=1e-6, warm_start=True, verbose=True)

    return p_var.value, v_var.value, prb.value


def converged(f_prev, f_next):
    return abs(f_next - f_prev) < tol


def analyze(n_w):
    try:
        p_, v_ = setup(n_w)
        c, f_prev, f_next, lagr_values = 0, 0.0, 0.0, {}

        while c < conf or not converged(f_prev, f_next):
            f_prev = f_next
            c += 1 if converged(f_prev, f_next) else -c

            # noinspection PyTypeChecker
            p, v, f_next = solve(p_, v_, n_w)

            lagr_values[time.monotonic()] = f_next
            tf.compat.v1.assign(p_, p, validate_shape=True, use_locking=True)
            tf.compat.v1.assign(v_, v, validate_shape=True, use_locking=True)

        with open('../perf-logs/sca-convergence.log', 'w') as f:
            json.dump(lagr_values, f)
        return True
    except Exception as e:
        print('[ERROR] SCAConvergence analyze: Exception caught while analyzing the convergence properties '
              'of the Successive Convex Approximation algorithm - {}'.format(traceback.print_tb(e.__traceback__)))
        return False


# Run Trigger
if __name__ == '__main__':
    print(f'[INFO] SCAConvergence main: Successive Convex Approximation | Data Length = {data_len} Mb | '
          f'UAV Average Power Constraint = {p_avg} Watts | Convergence Analysis Status = {analyze(256)}')
