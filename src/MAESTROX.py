"""
MAESTRO-X: Multiscale Adaptive Energy-conscious Scheduling and TRajectory Optimization (eXtended)

This script encapsulates the operations of our multiagent heuristics -- namely, command-and-control network design,
spread maximization, consensus-driven conflict resolution, piggybacking, frequency reuse, and M/G/x queuing dynamics
for channel & transceiver service modeling -- for orchestrating a swarm of rotary-wing Unmanned Aerial Vehicles (UAVs)
serving as cellular relays (Decode-and-Forward) to augment the service and coverage capabilities of a Base Station (BS).

Reference Paper for the associated mathematical modeling:

    @misc{https://doi.org/10.48550/arxiv.2209.07655,
          doi = {10.48550/ARXIV.2209.07655},
          url = {https://arxiv.org/abs/2209.07655},
          author = {Keshavamurthy, Bharath and Michelusi, Nicolo},
          title = {Multiscale Adaptive Scheduling and Path-Planning for Power-Constrained UAV-Relays via SMDPs},
          publisher = {arXiv},
          year = {2022},
          copyright = {Creative Commons Attribution 4.0 International}
    }

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

import uuid
import ntplib
import plotly
import numpy as np
import traceback as tb
import tensorflow as tf
from threading import Lock
from datetime import datetime
from typing import List, Tuple
from collections import namedtuple
from scipy.optimize import minimize
from simpy import Environment, Resource
from dataclasses import field, dataclass
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Simulation parameters 
"""
pi = np.pi
np.random.seed(6)
agent_id = '1221d753-0bd1-477f-96c6-10899725037b.log'
ip_dir, op_dir = '../logs/policies/', '../logs/evaluations/'
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)
plotly.tools.set_credentials_file(username='bkeshav1', api_key='PUYaTVhV1Ok04I07S4lU')
u0, u1, u2, th_c_min, th_c_max = [0.0, 500.0], [400.0, -300.0], [-400.0, -300.0], 0.0, 2 * pi
ntp_client, ntp_url, policy_file = ntplib.NTPClient(), 'pool.ntp.org', ''.join([ip_dir, agent_id])
a, bw, n_c, n_u, n_g, n_r, n_x_bs, n_x_u, z_1, z_2 = 1e3, 20e6, 4, 1, 30, int(1e4), 10, 3, 9.61, 0.16
utip, v0, p1, p2, p3, v_min, v_max, th_min, th_max = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0, 0.0, 5.5
n_w, p_av, bw_, ap, ap_, kp, p_len, ld, k_1, k_2 = 1024, 1.2e3, bw / n_c, 2.0, 2.8, 0.2, 10e6, 3.33e-3, 1.0, 0.0512
r_p, ra_tol, ra_conf, fr_tol, gm_, h_bs, h_u, h_gn = 1e-2, 1e-10, 10, 40.0, linear((5e6 * 40) / bw_), 80.0, 200.0, 0.0
nu, a_o, r_num, th_num, d_00, d_01, d_c, th_c_num = 0.99 / p_av, 50.0, 25, 25, 1.0, -np.log(0.93) / ld, 1e-10, int(1e4)

"""
DTOs
"""

' Environment '
users = [uuid.uuid4() for _ in range(n_g)]
channels = [uuid.uuid4() for _ in range(n_c)]
serv_nodes = [uuid.uuid4() for _ in range(n_u + 1)]
uavs = [tf.Variable([u0], dtype=tf.float64), tf.Variable([u1], dtype=tf.float64), tf.Variable([u2], dtype=tf.float64)]

' Transient collections '
r_us = np.linspace(a_o, a - a_o, r_num, dtype=np.float64)
th_us = np.linspace(th_min, th_max, th_num, dtype=np.float64)
th_cs = np.linspace(th_c_min, th_c_max, th_c_num, dtype=np.float64)

' Mappings for middleware publish-subscribe '
trx_assoc = namedtuple('trx_assoc', ['busy', 'user', 'ch_assoc'])
ch_assoc = namedtuple('ch_assoc', ['current', 'decode', 'forward', 'is_reuse'])
serv_cost = namedtuple('serv_cost', ['serv_time', 'lagr', 'ch_wait', 'trx_wait'])

' Mappings for rate-adapted link performance analyses '
service_cost = namedtuple('service_cost', ['t', 'e', 'x'])
phase_cost = namedtuple('phase_cost', ['t', 't_p', 'e', 'e_p'])

' Metrics to be analyzed '
args, serv_times, wait_times = [], [], []


@dataclass(order=True)
class ControlFrame:
    """
    A control frame skeleton
    """

    users: List
    seq_num: int
    node_id: uuid
    node_type: str
    state_flag: str
    frame_type: str
    transceivers: List
    cost_estimate: Tuple
    trajectory: tf.Variable
    current_position: tf.Variable
    reporting_period: float = r_p
    num_trx: int = field(init=False)
    timestamp: datetime = datetime.utcfromtimestamp(ntp_client.request(ntp_url).tx_time)

    def __post_init__(self):
        self.num_trx = n_x_u if self.node_type == 'UAV' else n_x_bs

        assert self.seq_num >= 0
        assert self.node_id in serv_nodes
        assert self.node_type in ['BS', 'UAV']
        assert len(self.transceivers) <= self.num_trx
        assert self.state_flag in ['OFF', 'WAIT', 'COMM', 'FAULT']


class ServiceNode(object):
    """
    A service node (BS/UAV) instance
    """

    def __init__(self, uuid_, type_, current_position_, trajectory_):
        self.id = uuid_
        self.type = type_
        self.cf_seq_num = 0
        self.state_flag = 'OFF'

        self.lock = Lock()
        self.running = True
        self.mw_registered = False

        self.trajectory = trajectory_
        self.current_position = current_position_

        self.num_trx = n_x_u if self.type == 'UAV' else n_x_bs

        self.users = []
        self.transient_users = []
        self.env_nodes = {_node: None for _node in serv_nodes}
        self.transceivers = {_trx: None for _trx in self.num_trx}

    def register(self):
        if not self.mw_registered:
            with self.lock:
                middleware.register(self)
                self.mw_registered = True

    def send(self, frame_type, user_id=None, cost_estimate=None):
        with self.lock:
            if frame_type == 'CTL':
                self.transient_users.remove(user_id)
                middleware.update(self, ControlFrame(self.users, self.cf_seq_num, self.id,
                                                     self.type, self.state_flag, 'CTL', self.transceivers,
                                                     cost_estimate, self.trajectory, self.current_position))
            elif frame_type == 'REQ':
                self.transient_users.append(user_id)
                middleware.update(self, ControlFrame(self.transient_users, self.cf_seq_num, self.id,
                                                     self.type, self.state_flag, 'REQ', self.transceivers,
                                                     cost_estimate, self.trajectory, self.current_position))
            else:
                raise NotImplementedError(f'[ERROR] [{self.id}] ServiceNode send: Unknown frame_type [{frame_type}]')

    def receive(self, src_id, cf):
        frame_type = cf.frame_type

        with self.lock:
            if frame_type == 'CTL':
                self.env_nodes[src_id] = cf
            elif frame_type == 'REQ':
                [self.transient_users.append(_u) for _u in cf.users if _u not in self.transient_users]
            else:
                raise NotImplementedError(f'[ERROR] [{self.id}] ServiceNode receive: Unknown frame_type [{frame_type}]')

    def spread_maximization(self):
        f, x = self.state_flag, self.current_position
        r_u, th_u = tf.norm(x), tf.math.atan(x[1] / x[0])
        i_ = np.random.choice([_ for _ in range(th_c_num)])
        i = min([_ for _ in range(r_num)], key=lambda _x: abs(r_u - r_us[_x]))
        peers = {_k: _v.current_position for _k, _v in self.env_nodes.items() if _v.state_flag == 'WAIT'}

        v_r = o_star(o_star[tf.argmin(s_wait - r_u)])

        def vel_fn(th_c):
            return (v_r ** 2 + (r_u * th_c) ** 2) ** 0.5

        def obj_fn(th_c):
            return nu * (mobility_pwr(vel_fn(th_c)) - p_av) * d_01

        constraints = ({'type': 'ineq', 'fun': lambda th_c: v_max - vel_fn(th_c)})
        th_c_star = abs(minimize(obj_fn, th_cs[i_], method='SLSQP', constraints=constraints, tol=d_c).x[0])

        if v_r > 0.0:
            r_nxt, th_nxt = r_us[i + 1], th_u + th_c_star * d_00
        elif v_r < 0.0:
            r_nxt, th_nxt = r_us[i - 1], th_u - th_c_star * d_00
        else:
            min_peer = min(peers.values, lambda _o: abs(th_u - tf.math.atan(_o[1] / _o[0])))

            th_peer = tf.math.atan(min_peer[1] / min_peer[0])
            sgn = min([1, -1], lambda _sgn: abs((th_u + _sgn * th_c_star * d_00) - th_peer))

            r_nxt, th_nxt = r_us[i], th_u + th_u + sgn * th_c_star * d_00

        with self.lock:
            self.current_position = tf.Variable([r_nxt * np.cos(th_nxt), r_nxt * np.sin(th_nxt)], dtype=tf.float64)

    @staticmethod
    def direct(x_gn):
        d_gb = np.sqrt(np.add(np.square(h_bs), np.square(np.linalg.norm(x_gn.numpy()))))
        phi_gb = np.arcsin(np.divide(h_bs, d_gb))

        r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

        link_perf.ra_tgpt(d_gb, phi_gb, r_los_gb, r_nlos_gb)

        phi_degrees_gb = (180.0 / np.pi) * phi_gb

        p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
        p_nlos_gb = 1 - p_los_gb

        r_bar_gb = np.add(np.multiply(p_los_gb, r_los_gb.numpy()), np.multiply(p_nlos_gb, r_nlos_gb.numpy()))

        return service_cost(t=p_len / r_bar_gb, e=0.0, x=None)

    @staticmethod
    def decode(x_gn, d_p, d_v, d_lp):
        r_gu = tf.norm(tf.subtract(d_p, tf.tile(tf.expand_dims(x_gn, axis=0), multiples=[d_p.shape[0], 1])), axis=1)
        h_uav = tf.constant(h_u, shape=r_gu.shape, dtype=tf.float64)

        d_gu = tf.sqrt(tf.add(tf.square(r_gu), tf.square(h_uav)))
        phi_gu = tf.asin(tf.divide(h_uav, d_gu))

        r_los_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_gu = tf.Variable(tf.zeros(shape=r_gu.shape, dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=n_w) as executor:
            for _i, (_d_gu, _phi_gu) in enumerate(zip(d_gu.numpy(), phi_gu.numpy())):
                executor.submit(link_perf.ra_tgpt, _d_gu, _phi_gu, r_los_gu[_i], r_nlos_gu[_i])

        phi_degrees_gu = (180.0 / pi) * phi_gu

        p_los_gu = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gu - z_1))))
        p_nlos_gu = tf.subtract(tf.ones(shape=p_los_gu.shape, dtype=tf.float64), p_los_gu)

        r_bar_gu = tf.add(tf.multiply(p_los_gu, r_los_gu), tf.multiply(p_nlos_gu, r_nlos_gu))

        t = tf.divide(tf.norm(tf.roll(d_p, shift=-1, axis=0)[:-1, :] - d_p[:-1, :], axis=1),
                      tf.where(tf.equal(d_v[:-1], 0.0), tf.ones_like(d_v[:-1]), d_v[:-1]))

        h_xtra = p_len - d_lp - tf.reduce_sum(tf.multiply(t, r_bar_gu[:-1]))

        t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_gu[-1]) if r_bar_gu[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
        e_p = (lambda: 0.0, lambda: (min_pwr * t_p))[h_xtra.numpy() > 0.0]()

        return phase_cost(t=tf.reduce_sum(t), t_p=t_p,
                          e=tf.reduce_sum(tf.multiply(t, pwr_cost(d_v[:-1], n_w))), e_p=e_p)

    @staticmethod
    def forward(f_p, f_v, f_lp):
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

        h_xtra = p_len - f_lp - tf.reduce_sum(tf.multiply(t, r_bar_ub[:-1]))

        t_p = (lambda: 0.0, lambda: (h_xtra / r_bar_ub[-1]) if r_bar_ub[-1] != 0.0 else np.inf)[h_xtra.numpy() > 0.0]()
        e_p = (lambda: 0.0, lambda: (min_pwr * t_p))[h_xtra.numpy() > 0.0]()

        return phase_cost(t=tf.reduce_sum(t), t_p=t_p,
                          e=tf.reduce_sum(tf.multiply(t, pwr_cost(f_v[:-1], n_w))), e_p=e_p)

    def service(self, x_gn, *args_):
        if self.type == 'UAV':
            d_p, d_v, d_lp, f_p, f_v, f_lp = args_
            t_ub, t_p_ub, e_ub, e_p_ub = self.forward(f_p, f_v, f_lp)
            t_gu, t_p_gu, e_gu, e_p_gu = self.decode(x_gn, d_p, d_v, d_lp)
            time, energy, end_position = t_gu + t_p_gu + t_ub + t_p_ub, e_gu + e_p_gu + e_ub + e_p_ub, f_p[-1]
        else:
            time, energy, end_position = self.direct(x_gn)

        return service_cost(t=time, e=energy, x=end_position)

    def compute_cost(self):
        with self.lock:
            ch_w_time, chn_assoc = self.ch_wait()
            trx_w_time, trxr_assoc = self.trx_wait()

            for tr_user in self.transient_users:
                x_gn = gn_map[tr_user]

                if self.type == 'UAV':
                    x_u, _wp, _vel, lp, lp_, _wpp, _velp = self.model(x_gn, ch_w_time + trx_w_time)

                    st = tf.argmin(tf.reduce_sum(tf.norm(s_comm - tf.concat(x_u, x_gn, axis=1), axis=1), axis=1))

                    if lp_ >= p_len:
                        _mpp = int(_wpp.shape[0] / 2)
                        s_time = self.service(x_gn, _wpp[:_mpp], _velp[:_mpp], 0.0, _wpp[_mpp:], _velp[_mpp:], 0.0)
                    else:
                        wp_, vel_ = p_star[st], v_star[st]
                        wp, vel = tf.concat([_wp, wp_], axis=0), tf.concat([_vel, vel_], axis=0)

                        mp = int(wp.shape[0] / 2)
                        s_time = self.service(x_gn, wp[:mp], vel[:mp], lp, wp[mp:], vel[mp:], lp_)

                else:
                    s_time = self.service(x_gn)

                self.send('CTL', tr_user, ch_w_time + trx_w_time + s_time)

    def expunge(self):
        if self.mw_registered:
            with self.lock:
                middleware.expunge(self)


class MessagingMiddleware(object):
    """
    Singleton | A messaging middleware framework for publish-subscribe
    """

    __msg_mw = None

    @staticmethod
    def msg_mw():
        if MessagingMiddleware.__msg_mw is None:
            MessagingMiddleware()
        return MessagingMiddleware.__msg_mw

    def __init__(self):
        self.nodes = {}
        self.lock = Lock()
        self.running = True

    def register(self, node):
        if self.running and node.id in serv_nodes:
            with self.lock:
                self.nodes[node.id] = node
            return True

        return False

    def update(self, src_node, cf):
        k = src_node.id

        if self.running:
            with self.lock:
                dst_nodes = [_v for _k, _v in self.nodes.items() if _k != k]
                self.notify(dst_nodes, k, cf)

    def notify(self, dst_nodes, src_id, cf):
        if self.running:
            with self.lock:
                [node.receive(src_id, cf) for node in dst_nodes]

    def expunge(self, node):
        if self.running and node.id in serv_nodes:
            with self.lock:
                del self.nodes[node.id]
            return True

        return False

    def stop(self):
        with self.lock:
            self.running = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


"""
Utilities
"""


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


@tf.function
@tf.autograph.experimental.do_not_convert
def pwr_cost(v):
    return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


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
        return ((2.0 ** (gm / bw_)) - 1.0) / (gm_ * 1.0 if los else kp * (d ** -ap if los else -ap_))

    def los_tgpt(self, d, phi, r_los):
        k = k_1 * np.exp(k_2 * phi)
        ppf_pt = rice.ppf(0.9999999999, k) ** 2.0
        df, nc, y = 2, (2 * k), (k + 1.0) * (1.0 / (gm_ * (d ** -ap)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * gm_ * (d ** -ap))))

        tf.compat.v1.assign(r_los, self.f_z(z_star), use_locking=True)

    def nlos_tgpt(self, d, r_nlos):
        ppf_pt = rayleigh.ppf(0.9999999999) ** 2.0
        df, nc, y = 2, 0, 1.0 / (gm_ * (kp * (d ** -ap_)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * gm_ * kp * (d ** -ap_))))

        tf.compat.v1.assign(r_nlos, self.f_z(z_star), use_locking=True)

    def ra_tgpt(self, d, phi, r_los, r_nlos):
        with ThreadPoolExecutor(n_w) as executor:
            executor.submit(self.nlos_tgpt, d, r_nlos)
            executor.submit(self.los_tgpt, d, phi, r_los)

    def avg_tgpt(self, d_s, phi_s):
        r_los_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)
        r_nlos_s = tf.Variable(tf.zeros(shape=d_s.shape, dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(n_w) as executor:
            for i, (d, phi) in enumerate(zip(d_s, phi_s)):
                executor.submit(self.ra_tgpt, d, phi, r_los_s[i], r_nlos_s[i])

        phi_degrees = (180.0 / np.pi) * phi_s

        p_los = 1.0 / (1.0 + (z_1 * tf.exp(-z_2 * (phi_degrees - z_1))))
        p_nlos = tf.subtract(tf.ones(shape=p_los.shape, dtype=tf.float64), p_los)

        return r_los_s, r_nlos_s, tf.add(tf.multiply(p_los, r_los_s), tf.multiply(p_nlos, r_nlos_s))


"""
Core operations
"""

min_pwr = mobility_pwr(22.0)
link_perf = LinkPerformance()
middleware = MessagingMiddleware()
x_bs = tf.constant([0.0, 0.0], dtype=tf.float64)

try:

    with open(policy_file, 'r') as file:
        args.append(tf.convert_to_tensor(file.readline().strip(), dtype=tf.string))
        for line in file.readlines():
            args.append(tf.convert_to_tensor(line.strip(), dtype=tf.float64))

except Exception as e:
    print(f'[ERROR] MAESTROX | Exception caught while parsing {policy_file}: {tb.print_tb(e.__traceback__)}.')

bs_delta_s, bs_energy_s, uav_delta_s, uav_energy_s, p_star, v_star, xi_star, o_star, u_star = args[9:]
uid, ell, p_avg, s_wait, a_wait, s_comm, a_comm, delta_s, energy_s = args[:9]
env, n_wait, n_comm = Environment(), s_wait.shape[0], s_comm.shape[0]

assert uid == agent_id and p_len == ell and p_av == p_avg

ch_resources = [Resource(env) for _ in range(n_c)]

x_gns = s_comm[:, 1, :].numpy()
gn_map = {users[_i]: x_gns[_i] for _i in range(n_g)}
