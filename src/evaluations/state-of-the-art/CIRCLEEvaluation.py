"""
This script evaluates the performance of the CIRCLE UAV trajectory heuristic from the state-of-the-art.

Reference Papers:

    @ARTICLE{CIRCLE_1,
    author={Hu, Qiyu and Cai, Yunlong and Liu, An and Yu, Guanding and Li, Geoffrey Ye},
    journal={IEEE Transactions on Wireless Communications},
    title={Low-Complexity Joint Resource Allocation and Trajectory Design for UAV-Aided Relay Networks With the
           Segmented Ray-Tracing Channel Model},
    year={2020},
    volume={19},
    number={9},
    pages={6179-6195},
    doi={10.1109/TWC.2020.3000864}}

    @ARTICLE{CIRCLE_2,
        author={Wang, Liang and Wang, Kezhi and Pan, Cunhua and Xu, Wei and Aslam, Nauman and Hanzo, Lajos},
        journal={IEEE Transactions on Cognitive Communications and Networking},
        title={Multi-Agent Deep Reinforcement Learning-Based Trajectory Planning for Multi-UAV Mobile Edge Computing},
        year={2021},
        volume={7},
        number={1},
        pages={73-84},
        doi={10.1109/TCCN.2020.3027695}}

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
import warnings
import numpy as np
import tensorflow as tf
from collections import namedtuple
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor
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
np.random.seed(6)
bw, n_c = 20e6, 4
n_xu, n_xb = 3, 10
pi, bw_ = np.pi, bw / n_c
snr_0 = linear((5e6 * 40) / bw_)
a, m, n, omi = 1e3, 256, 30, 1.0
a_los, a_nlos, kappa = 2.0, 2.8, 0.2
r_bounds, th_bounds, h_bs, h_uavs, h_gns = (-a, a), (0, 2 * pi), 80.0, 200.0, 0.0
k_1, k_2, z_1, z_2, ra_conf, ra_tol = 1.0, np.log(100) / 90.0, 9.61, 0.16, 10, 1e-10
utip, v0, p1, p2, p3, v_min, v_max, v_num = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0, 25
data_lens, arr_rates, num_uavs, num_gns = [1e6, 10e6, 100e6], {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}, 1, n

"""
Configurations-III: Deployment settings
"""

x_bs = tf.constant([0.0, 0.0], dtype=tf.float64)

r_gns, th_gns = uniform(0, a ** 2, num_gns) ** 0.5, uniform(0, 2 * pi, num_gns)
x_gns = tf.constant(list(zip(r_gns * np.cos(th_gns), r_gns * np.sin(th_gns))), dtype=tf.float64)

r_uavs = [500.0]  # r_uavs (2 UAVs) = [333.33, 666.67] | r_uavs (3 UAVs) = [250.0, 500.0, 750.0]

th_uavs = [np.linspace(0, 2 * pi, m) for _ in range(num_uavs)]
x_uavs = tf.constant([list(zip(r_uavs[_u] * np.cos(th_uavs[_u]),
                               r_uavs[_u] * np.sin(th_uavs[_u]))) for _u in range(num_uavs)], dtype=tf.float64)
v_uavs = tf.constant([choice(np.linspace(v_min, v_max, v_num), size=m) for _ in range(num_uavs)], dtype=tf.float64)

"""
Utilities
"""


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


def wait(u, x_u, z_u):
    """
    Waiting dynamics
    """
    t_u = time.monotonic() - z_u
    r_u, th_u = tf.norm(x_u), tf.atan(x_u[1] / x_u[0])
    d_u = tf.multiply(tf.gather(v_uavs[u], tf.where(tf.equal(x_uavs[u], x_u))[0, 0]), t_u)
    return x_uavs[u, tf.argmin(d_u - tf.abs(tf.atan(x_uavs[u, :, 1] / x_uavs[u, :, 0]) - th_u) * r_u).numpy(), :]


def gn_request(env, r, xs, ell, chs, w, s, z, e, n_w):
    """
    SimPy queueing model: GN request
    """
    [tf.compat.v1.assign(xs[u], wait(u, xs[u], z[u]), validate_shape=True,
                         use_locking=True) if z[u] > 0.0 else None for u in range(num_uavs)]

    arr = env.now
    s_u, s_t, s_e, s_x = min([aggregated_service_metrics(u, xs[u], x_gns[r], ell, n_w) for u in range(num_uavs)],
                             key=lambda x: omi * x.t + (1 - omi) * x.e + len(chs[x.u].put_queue) + len(chs[x.u].users))

    with chs[s_u].request() as req:
        yield req
        w.append(env.now - arr)
        s.append(s_t)
        e.append(s_e)
        yield env.timeout(s_t)

    z[s_u] = time.monotonic()
    tf.compat.v1.assign(xs[s_u], s_x, validate_shape=True, use_locking=True)


def arrivals(env, xs, chs, n_r, ell, arr, w, s, e, n_w):
    """
    SimPy queueing model: Poisson arrivals
    """
    z = {u: 0.0 for u in range(num_uavs)}
    for r in range(n_r):
        env.process(gn_request(env, r, xs, ell, chs, w, s, z, e, n_w))
        yield env.timeout(-np.log(random_sample()) / arr)


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

        tf.compat.v1.assign(r_los, self.__f_z(z_star), validate_shape=True, use_locking=True)

    def __nlos_throughput(self, d, r_nlos):
        df, nc, y = 2, 0, 1 / (snr_0 * (kappa * (d ** -a_nlos)))

        z_star = self.__bisect(self.__f, df, nc, y, 0,
                               self.__z(bw_ * np.log2(1 + (
                                       rayleigh.ppf(0.9999999999) ** 2) * snr_0 * kappa * (d ** -a_nlos))))

        tf.compat.v1.assign(r_nlos, self.__f_z(z_star), validate_shape=True, use_locking=True)

    def adapted_throughput(self, d, phi, r_los, r_nlos, n_w):
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            executor.submit(self.__nlos_throughput, d, r_nlos)
            executor.submit(self.__los_throughput, d, phi, r_los)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'[INFO] LinkPerformance Termination: Tearing things down - '
              f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {exc_tb}.')


"""
Core operations
"""

lpf = LinkPerformance()
hover_pwr = mobility_pwr(0.0)

gu_metrics = namedtuple('gu_metrics', ['t', 't_p', 'e', 'e_p'])
ub_metrics = namedtuple('gu_metrics', ['t', 't_p', 'e', 'e_p'])
service_metrics = namedtuple('service_metrics', ['u', 't', 'e', 'x'])


def gu_subtrajectory(u, m_, x_gn):
    m__ = tf.argmin(tf.sqrt(tf.add(tf.square(tf.norm(x_uavs[u] - tf.tile(tf.expand_dims(x_gn, axis=0),
                                                                         multiples=[m, 1]), axis=1)),
                                   tf.square(tf.constant(abs(h_uavs - h_gns), shape=(m,), dtype=tf.float64)))))

    return {'v': v_uavs[u, m_:m__ + 1] if m_ < m__ else v_uavs[u, m__:m_ + 1],
            'p': x_uavs[u, m_:m__ + 1, :] if m_ < m__ else x_uavs[u, m__:m_ + 1, :]}


def ub_subtrajectory(u, m_):
    m__ = tf.argmin(tf.sqrt(tf.add(tf.square(tf.norm(x_uavs[u] - tf.tile(tf.expand_dims(x_bs, axis=0),
                                                                         multiples=[m, 1]), axis=1)),
                                   tf.square(tf.constant(abs(h_uavs - h_bs), shape=(m,), dtype=tf.float64)))))

    return {'v': v_uavs[u, m_:m__ + 1] if m_ < m__ else v_uavs[u, m__:m_ + 1],
            'p': x_uavs[u, m_:m__ + 1, :] if m_ < m__ else x_uavs[u, m__:m_ + 1, :]}


@tf.function
@tf.autograph.experimental.do_not_convert
def pwr_cost(v, n_w):
    return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


def gu_link_metrics(v, p, x_gn, ell, n_w):
    r_gu = tf.norm(tf.subtract(p, tf.tile(tf.expand_dims(x_gn, axis=0), multiples=[p.shape[0], 1])), axis=1)
    h_uav = tf.constant(h_uavs, shape=r_gu.shape, dtype=tf.float64)
    d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_uav)))
    phi_gu = tf.asin(tf.divide(h_uav, d_gu))

    r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
    r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

    with ThreadPoolExecutor(max_workers=n_w) as executor:
        for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu.numpy(), phi_gu.numpy())):
            executor.submit(lpf.adapted_throughput, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i], n_w)

    phi_degrees_gu = (180.0 / pi) * phi_gu
    p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
    p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)
    r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0),
                                                                                              tf.ones_like(v[:-1]),
                                                                                              v[:-1]))

    h_xtra = ell - tf.reduce_sum(tf.multiply(t, r_bar_gu[:-1]))

    t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
    e_p = (lambda: 0.0, lambda: (hover_pwr * t_p))[h_xtra.numpy() > 0.0]()

    return gu_metrics(t=tf.reduce_sum(t), t_p=t_p, e=tf.reduce_sum(tf.multiply(t, pwr_cost(v[:-1], n_w))), e_p=e_p)


def ub_link_metrics(v, p, ell, n_w):
    r_ub = tf.norm(tf.subtract(p, tf.tile(tf.expand_dims(x_bs, axis=0), multiples=[p.shape[0], 1])), axis=1)
    h_ub = tf.constant(abs(h_uavs - h_bs), shape=r_ub.shape, dtype=tf.float64)
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

    t = tf.divide(tf.norm(tf.roll(p, shift=-1, axis=0)[:-1, :] - p[:-1, :], axis=1), tf.where(tf.equal(v[:-1], 0.0),
                                                                                              tf.ones_like(v[:-1]),
                                                                                              v[:-1]))

    h_xtra = ell - tf.reduce_sum(tf.multiply(t, r_bar_ub[:-1]))

    t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
    e_p = (lambda: 0.0, lambda: (hover_pwr * t_p))[h_xtra.numpy() > 0.0]()

    return ub_metrics(t=tf.reduce_sum(t), t_p=t_p, e=tf.reduce_sum(tf.multiply(t, pwr_cost(v[:-1], n_w))), e_p=e_p)


def aggregated_service_metrics(u, x_uav, x_gn, ell, n_w):
    u_idx = tf.where(tf.equal(x_uavs[u], x_uav))[0, 0]
    v_gu, p_gu = gu_subtrajectory(u, u_idx, x_gn).values()
    t_gu, t_p_gu, e_gu, e_p_gu = gu_link_metrics(v_gu, p_gu, x_gn, ell, n_w)

    u_idx = tf.where(tf.equal(x_uavs[u], p_gu[-1]))[0, 0]

    v_ub, p_ub = ub_subtrajectory(u, u_idx).values()
    t_ub, t_p_ub, e_ub, e_p_ub = ub_link_metrics(v_ub, p_ub, ell, n_w)

    return service_metrics(u=u, t=t_gu + t_p_gu + t_ub + t_p_ub, e=e_gu + e_p_gu + e_ub + e_p_ub, x=p_ub[-1])


# noinspection PyTypeChecker
def evaluate(n_w):
    for data_len in data_lens:
        env = Environment()
        wait_times, serv_times, serv_energies, total_times = [], [], [], []

        env.process(arrivals(env, tf.Variable([x_uavs[u, 0, :] for u in range(num_uavs)], dtype=tf.float64),
                             [Resource(env) for _ in range(num_uavs)], n, data_len, arr_rates[data_len],
                             wait_times, serv_times, serv_energies, n_w))

        env.run()

        comm_serv_times, queue_wait_times = np.array(serv_times), np.array(wait_times)

        avg_serv_time, avg_wait_time = np.mean(comm_serv_times), np.mean(queue_wait_times)
        avg_total_time = np.mean(comm_serv_times + queue_wait_times)

        comm_serv_energies = np.array(serv_energies)
        avg_serv_power = np.mean(np.divide(comm_serv_energies, comm_serv_times))

        print(f'[INFO] CIRCLEEvaluation evaluate: UAVs = {num_uavs} | GNs/Requests = {num_gns} | '
              f'Data Length = {data_len / 1e6} Mb | Average Power Consumption = {avg_serv_power / 1e3} kW | '
              f'Comm = {avg_serv_time} seconds | Wait = {avg_wait_time} seconds | Total = {avg_total_time} seconds.')


# Run Trigger
if __name__ == '__main__':
    evaluate(1024)
