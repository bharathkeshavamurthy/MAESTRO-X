"""
This paper evaluates the operational performance of the Successive Convex Approximation (SCA) based Path Discretization
framework in the state-of-the-art. We incorporate the channel model, UAV power & mobility model, and DLL dynamics from
our paper in this SCA framework's operational/functional performance evaluation to make all the comparisons fair.

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
import traceback
import numpy as np
import cvxpy as cp
import tensorflow as tf
from numpy.linalg import norm
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from scipy.interpolate import UnivariateSpline
from numpy.random import choice, uniform, random_sample

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
pi = np.pi
bw_ = 2.5e6
np.random.seed(6)
a, m, m_ip, n = 1e3, 14, 2, 30
snr_0 = linear((5e6 * 40) / bw_)
a_los, a_nlos, kappa, m_post = 2.0, 2.8, 0.2, m_ip * (m + 2)
r_bounds, th_bounds, h_uavs, h_gns = (-a, a), (0, 2 * pi), 200.0, 0.0
max_iters, eps_abs, eps_rel, warm_start, verbose = int(1e6), 1e-6, 1e-6, True, True
utip, v0, p1, p2, p3, v_min, v_max, v_num = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0, 10
data_len, arr_rates, num_uavs, num_gns = [1e6, 10e6, 100e6][1], {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}, 2, n
pc, k_1, k_2, z_1, z_2, conf, tol, cvx_conf, cvx_tol = 90.0, 1.0, np.log(100) / 90.0, 9.61, 0.16, 3, 1e-3, 1, 1000.0

"""
Configurations-III: Deployment parameters
"""

r_gns, th_gns = uniform(0, a ** 2, num_gns) ** 0.5, uniform(0, 2 * pi, num_gns)
x_gns = tf.constant(list(zip(r_gns * np.cos(th_gns), r_gns * np.sin(th_gns))), dtype=tf.float64)

x_uavs = [tf.tile(tf.expand_dims(tf.constant([0.0, 500.0], dtype=tf.float64), axis=0), multiples=[num_gns, 1]),
          tf.tile(tf.expand_dims(tf.constant([-400.0, -300.0], dtype=tf.float64), axis=0), multiples=[num_gns, 1]),
          tf.tile(tf.expand_dims(tf.constant([400.0, -300.0], dtype=tf.float64), axis=0), multiples=[num_gns, 1])]

"""
Utilities
"""


def obj_fn(t, tau, y, delta):
    """
    UAV mobility energy consumption
    """
    return p3 * np.sum(np.divide(delta ** 3.0, t ** 2), axis=0) + \
        pc * np.sum(np.sum(tau, axis=1)) + p2 * np.sum(y, axis=0) + \
        p1 * np.sum(np.add(t, np.divide(3.0 * delta ** 2, np.multiply(utip ** 2.0, t))), axis=0)


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


def gn_request(env, r, chs, w, s):
    """
    SimPy queueing model: GN request
    """
    arr = env.now
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        w.append(env.now - arr)
        yield env.timeout(s[r])


def arrivals(env, chs, n_r, arr, w, s):
    """
    SimPy queueing model: Poisson arrivals
    """
    for r in range(n_r):
        env.process(gn_request(env, r, chs, w, s))
        yield env.timeout(-np.log(random_sample()) / arr)


# noinspection PyMethodMayBeStatic
class RandomTrajectoriesGeneration(object):
    """
    Random trajectories generation
    """

    def __enter__(self):
        return self

    def generate(self):
        (r_min, r_max), (th_min, th_max) = r_bounds, th_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)
        return tf.concat([tf.multiply(r, tf.math.cos(theta)), tf.multiply(r, tf.math.sin(theta))], axis=1)

    def optimize(self, q_i, q_f, traj):
        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), (m_ip * len(i_s)), dtype=np.float64)

        return tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([q_i[:, 0], traj[:, 0], q_f[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([q_i[:, 1], traj[:, 1], q_f[:, 1]], axis=0), s=0)(x)))), a, axes=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}')


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
        return np.sqrt(2 * ((2 ** (gamma / bw_)) - 1))

    def __u(self, gamma, d, los):
        return ((2 ** (gamma / bw_)) - 1) / (snr_0 * 1 if los else kappa * (d ** -a_los if los else -a_nlos))

    def __los_throughput(self, d, phi):
        k = k_1 * np.exp(k_2 * phi)
        df, nc, y = 2, (2 * k), (k + 1) * (1 / (snr_0 * (d ** -a_los)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(bw_ * np.log2(1 + (rice.ppf(0.9999999999, k) ** 2) * snr_0 * (d ** -a_los))))

        return self.__f_z(z_star)

    def __nlos_throughput(self, d):
        df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -a_nlos)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(bw_ * np.log2(1 + (
                                       rayleigh.ppf(0.9999999999) ** 2) * snr_0 * kappa * (d ** -a_nlos))))

        return self.__f_z(z_star)

    def __adapted_throughput(self, d, phi):
        r_los, r_nlos = [], []

        for k in range(num_gns):
            r_nlos.append(self.__nlos_throughput(d[k]))
            r_los.append(self.__los_throughput(d[k], phi[k]))

        return r_los, r_nlos

    def average_throughputs(self, r_s, d_s, phi_s):
        r_los_s, r_nlos_s = [], []

        for i, (d, phi) in enumerate(zip(d_s, phi_s)):
            r_los, r_nlos = self.__adapted_throughput(d, phi)
            r_nlos_s.append(r_nlos)
            r_los_s.append(r_los)

        phi_d = (180.0 / np.pi) * phi_s
        r_los_s = tf.constant(r_los_s, dtype=tf.float64)
        r_nlos_s = tf.constant(r_nlos_s, dtype=tf.float64)
        p_los = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_d - z_1))))
        p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)
        tf.compat.v1.assign(r_s, tf.add(tf.multiply(p_los, r_los_s), tf.multiply(p_nlos, r_nlos_s)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformance Termination: Tearing things down - '
              f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {exc_tb}')


"""
Core operations
"""


def setup():
    q0 = tf.constant(tf.zeros(shape=(num_uavs, 2), dtype=tf.float64), dtype=tf.float64)
    q = tf.Variable(tf.zeros(shape=(num_uavs, m_post, 2), dtype=tf.float64), dtype=tf.float64)
    y = tf.Variable(tf.zeros(shape=(num_uavs, m_post - 1), dtype=tf.float64), dtype=tf.float64)
    t = tf.Variable(tf.zeros(shape=(num_uavs, m_post - 1), dtype=tf.float64), dtype=tf.float64)
    s = tf.Variable(tf.zeros(shape=(num_uavs, m_post - 1, num_gns), dtype=tf.float64), dtype=tf.float64)
    tau = tf.Variable(tf.zeros(shape=(num_uavs, m_post - 1, num_gns), dtype=tf.float64), dtype=tf.float64)
    v = tf.Variable(choice(np.linspace(v_min, v_max, v_num), size=[num_uavs, m_post - 1]), dtype=tf.float64)

    for u in range(num_uavs):
        with RandomTrajectoriesGeneration() as generator:
            tf.compat.v1.assign(q[u], generator.optimize(tf.expand_dims(x_uavs[u][0], axis=0),
                                                         tf.expand_dims(q0[u], axis=0), generator.generate()))

        tf.compat.v1.assign(t[u], tf.divide(tf.norm(tf.roll(q[u], -1, 0)[:-1, :] - q[u, :-1, :], axis=1),
                                            tf.where(tf.equal(v[u], 0.0), tf.ones_like(v[u]), v[u])))

        tf.compat.v1.assign(tau[u], tf.tile(tf.expand_dims(tf.divide(t[u], num_gns), axis=1), multiples=[1, num_gns]))

    return q, t, tau, y, s


# noinspection PyTypeChecker
def solve(u, qi, qf, *args):
    q, t, tau, y, s = (mem.numpy() for mem in setup())
    q_, y_, s_, r_ = (np.nan_to_num(arg.numpy()) for arg in args)

    q_var = cp.Variable((m_post, 2), value=q[u])
    t_var = cp.Variable((m_post - 1,), value=t[u])
    s_var = cp.Variable((m_post - 1, num_gns), value=s[u])
    y_var = cp.Variable((m_post - 1,), value=np.sqrt(y[u]))
    tau_var = cp.Variable((m_post - 1, num_gns), value=tau[u])
    v_var = cp.Variable((m_post - 1,), value=np.divide(norm(np.roll(q[u], -1, 0)[:-1, :] - q[u, :-1, :], axis=1), t[u]))

    delta = cp.norm(q_var[1:, :] - q_var[:-1, :], axis=1)
    delta_ = norm(np.roll(q_, -1, 0)[:-1, :] - q_[:-1, :], axis=1)

    s_[s_ < 0.0] = 0.0
    tmp_mul = 2 / (v0 ** 2)
    tmp_y1, tmp_y2, tmp_y3 = np.sqrt(y_), np.sqrt(y_) * 2.0, np.add(y_, np.multiply(-1 / (v0 ** 2), np.square(delta_)))

    nrg = p2 * cp.sum(y_var, axis=0) + \
          p3 * cp.sum(cp.multiply(t_var.value, cp.power(v_var, 3)), axis=0) + \
          p1 * cp.sum(t_var, axis=0) + pc * cp.sum(cp.sum(tau_var, axis=1), axis=0) + \
          p1 * cp.sum((3.0 * cp.multiply(t_var.value, cp.power(v_var, 2))) / (utip ** 2.0), axis=0)

    d_ct = cp.sum(s_ + cp.multiply(np.sqrt(s_) * 2.0, cp.sqrt(s_var) - np.sqrt(s_)), axis=0)
    s_ct = np.multiply(np.square(y_var.value), tmp_y3 + np.multiply(tmp_y2, y_var.value - tmp_y1) +
                       np.multiply(tmp_mul, np.sum(np.multiply(np.roll(q_, -1, 0)[:-1, :] - q_[:-1, :],
                                                               np.roll(q[u], -1, 0)[:-1, :] - q[u, :-1, :]), axis=1)))

    cts = [y_var >= 0.0, t_var >= 0.0, tau_var >= 0.01, delta <= cp.multiply(v_max, t_var),
           v_var >= 0.0, v_var <= v_max, q_var[0] == qi, cp.sum(tau_var, axis=1) <= t_var,
           q_var[-1] == qf, data_len <= d_ct, cp.power(t_var, 4) <= s_ct, s_var <= cp.multiply(r_[:-1], tau_var)]

    prb = cp.Problem(cp.Minimize(nrg), cts)
    prb.solve('SCS', max_iters=max_iters, eps_abs=eps_abs, eps_rel=eps_rel, warm_start=warm_start, verbose=verbose)

    print(f'UAV Index = {u} | Primal Value = {prb.value} | Problem Status = {prb.solution.status}')

    return q_var.value, t_var.value, tau_var.value, prb.value


def converged(f_prev, f_next):
    return abs(f_next - f_prev) < cvx_tol


def sca():
    nrgs = []
    q_, t_, tau_, y_, s_ = setup()
    q0 = tf.constant(tf.zeros(shape=(num_uavs, 2), dtype=tf.float64), dtype=tf.float64)
    h_ = tf.constant(abs(h_uavs - h_gns), shape=(num_uavs, m_post, num_gns), dtype=tf.float64)
    r_ = tf.Variable(tf.zeros(shape=(num_uavs, m_post, num_gns), dtype=tf.float64), dtype=tf.float64)

    for u in range(num_uavs):
        c, f_prev, f_next = 0, 0.0, 0.0

        while c < cvx_conf or not converged(f_prev, f_next):
            lperf = LinkPerformance()
            c, f_prev = c + 1 if converged(f_prev, f_next) else 0, f_next
            delta_ = tf.norm(tf.roll(q_[u], -1, 0)[:-1, :] - q_[u, :-1, :], axis=1)
            dxy = tf.Variable([tf.norm(x_gns - q_[u, _m], axis=1) for _m in range(m_post)], dtype=tf.float64)

            d = tf.sqrt(tf.add(tf.square(h_[u]), tf.square(dxy)))
            phi = tf.asin(tf.divide(h_[u], d))

            lperf.average_throughputs(r_[u], d, phi)
            tf.compat.v1.assign(s_[u], tf.multiply(tau_[u], r_[u][:-1]))

            tf.compat.v1.assign(y_[u],
                                tf.sqrt(tf.add(t_[u] ** 4.0,
                                               tf.divide(delta_ ** 4.0,
                                                         4.0 * v0 ** 4.0))) - tf.divide(delta_ ** 2.0, 2.0 * v0 ** 2.0))

            q, t, tau, f_next = solve(u, tf.squeeze(x_uavs[u][0]), q0[u], q_[u], y_[u], s_[u], r_[u])

            tf.compat.v1.assign(q_[u], q)
            tf.compat.v1.assign(t_[u], t)
            tf.compat.v1.assign(tau_[u], tau)

        nrgs.append(f_next)

    return q_, t_, tau_, r_, nrgs


def evaluate():
    q, t, tau, r, nrgs = sca()

    n_u, n_k = num_uavs, num_gns
    d_k = tf.Variable(tf.multiply(r[:, :-1, :], tau))
    [tf.compat.v1.assign(d_k[_u, _m, :], tf.multiply(d_k[_u, _m, :],
                                                     t[_u, _m])) for _u in range(n_u) for _m in range(m_post - 1)]

    dp = tf.subtract(tf.reduce_sum(tf.reduce_sum(d_k, axis=1), axis=0),
                     tf.constant(data_len, shape=(n_k,), dtype=tf.float64))
    serv_status = tf.less_equal(dp, tf.constant(tf.zeros(shape=(n_k,), dtype=tf.float64), dtype=tf.float64))

    tp = tf.constant(
        np.abs([dp[_k].numpy() /
                np.mean([tau[_u, -1, _k].numpy() *
                         r[_u, -1, _k].numpy() for _u in range(n_u)])
                if serv_status[_k] else 0.0 for _k in range(n_k)]), dtype=tf.float64)

    s_times = [tp[_k] + np.mean([
        tf.reduce_sum(tf.multiply(tau[_u, :, _k], t[_u, :])) for _u in range(n_u)]) for _k in range(n_k)]

    w_times, env = [], Environment()
    env.process(arrivals(env, [Resource(env) for _ in range(num_uavs)], n_k, arr_rates[data_len], w_times, s_times))

    env.run()

    avg_s_time = np.mean(s_times, axis=0)
    avg_w_time = np.mean(w_times, axis=0)
    avg_t_time = np.mean(np.add(s_times, w_times), axis=0)

    avg_pwr = np.mean([np.mean(np.divide(nrgs[_u], tf.reduce_sum(t[_u]).numpy())) for _u in range(n_u)]) + \
              np.nan_to_num(
                  tf.divide(tf.reduce_mean(tf.multiply(tp, mobility_pwr(0.0)), axis=0), tf.reduce_sum(tp)).numpy())

    print(f'[INFO] SCAEvaluation evaluate: UAVs = {n_u} | GNs/Requests = {n_k} | '
          f'Data Length = {data_len / 1e6} Mb | Average Power Consumption = {avg_pwr / 1e3} kW | '
          f'Comm Times = {avg_s_time} s | Wait Times = {avg_w_time} s | Total Times = {avg_t_time} s')


# Run Trigger
if __name__ == '__main__':
    evaluate()
