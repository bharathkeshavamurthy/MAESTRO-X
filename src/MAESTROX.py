"""
MAESTRO-X: Multiscale Adaptive Energy-conscious Scheduling and TRajectory Optimization (eXtended)

This script encapsulates the operations of our multi-agent heuristics -- namely, command-and-control network design,
spread maximization, consensus-driven conflict resolution, piggybacking, frequency reuse, and queuing dynamics --
for orchestrating a swarm of rotary-wing Unmanned Aerial Vehicles (UAVs) serving as cellular relays (D&F) to
augment the service and coverage capabilities of a terrestrial Base Station (BS).

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

import re
import uuid
import ntplib
import plotly
import numpy as np
import traceback as tb
import tensorflow as tf
from threading import Lock
from datetime import datetime
from dataclasses import dataclass
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from simpy import Environment, Resource
from scipy.stats import rice, ncx2, rayleigh
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Simulation parameters 
"""

pi = np.pi
np.random.seed(6)
a, n_u, n_r, n_w = 1e3, 3, int(1e4), 1024
agent_id = '1221d753-0bd1-477f-96c6-10899725037b'
depl_env, rf, le_l, le_m, le_h = 'rural', n_u, 1, 10, 100
ip_dir, op_dir = '../logs/policies/', '../logs/evaluations/'
plotly.tools.set_credentials_file(username='<User_Name>', api_key='<API_Key>')
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)
utip, v0, p1, p2, p3, v_min, v_max = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0
data_len, p_avg = [1e6, 10e6, 100e6][0], np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[0]
u0, u1, u2, th_c_min, th_c_max, th_c_num = [0.0, 500.0], [400.0, -300.0], [-400.0, -300.0], 0.0, 2 * pi, int(1e4)
ntp_client, ntp_url, policy_file = ntplib.NTPClient(), 'pool.ntp.org', f'{ip_dir}{int(data_len / 1e6)}-{int(p_avg)}.log'

arr_rates_r = {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}
arr_rates_l = {_k: _v * rf * le_l for _k, _v in arr_rates_r.items()}
arr_rates_m = {_k: _v * rf * le_m for _k, _v in arr_rates_r.items()}
arr_rates_h = {_k: _v * rf * le_h for _k, _v in arr_rates_r.items()}

'''
TODO: Change k_1, k_2, z_1, and z_2 according to the deployment environment
TODO: Change bw and n_c according to the deployment environment (Verizon LTE/LTE-A/5G)
'''
if depl_env == 'rural':
    bw, n_c, arr_rates = 10e6, 2, arr_rates_l
    k_1, k_2, z_1, z_2 = 1.0, np.log(100) / 90.0, 9.61, 0.16
elif depl_env == 'suburban':
    bw, n_c, arr_rates = 20e6, 4, arr_rates_m
    k_1, k_2, z_1, z_2 = 1.0, np.log(100) / 90.0, 9.61, 0.16
else:
    bw, n_c, arr_rates = 40e6, 8, arr_rates_h
    k_1, k_2, z_1, z_2 = 1.0, np.log(100) / 90.0, 9.61, 0.16

bw_ = bw / n_c
ap, ap_, kp, snr_0 = 2.0, 2.8, 0.2, linear((5e6 * 40) / bw_)
ld, r_p, ra_tol, ra_conf, h_bs, h_u, h_gn = arr_rates[data_len], 1e-2, 1e-10, 10, 80.0, 200.0, 0.0
n_l, min_dist, d_00, d_01, d_c, th_c_num, snr_deg_tol = 25, 25.0, 1.0, -np.log(0.93) / ld, 1e-10, int(1e4), linear(5.0)

"""
Configurations-III: Deployment settings
"""

''' BS deployment '''
x_bs = tf.constant([0.0, 0.0], dtype=tf.float64)

''' UAV(s) deployment '''

r_us = np.linspace(start=0.0, stop=a, num=n_l)
th_us = [np.linspace(start=0.0, stop=2 * pi, num=int((2 * pi * _r) / min_dist) + 1) for _r in r_us]

x_us = tf.concat([_r * np.einsum('ji', np.vstack([np.cos(th_us[_i]),
                                                  np.sin(th_us[_i])])) for _i, _r in enumerate(r_us)], axis=0)

''' GNs deployment (uniform-circular) '''

r_gns = np.linspace(start=0.0, stop=a, num=n_l)
th_gns = [np.linspace(start=0.0, stop=2 * pi, num=int((2 * pi * _r) / min_dist) + 1) for _r in r_gns]

x_gns = tf.concat([_r * np.einsum('ji', np.vstack([np.cos(th_gns[_i]),
                                                   np.sin(th_gns[_i])])) for _i, _r in enumerate(r_gns)], axis=0)

n_g = x_gns.shape[0]
gn_indices = [_ for _ in range(n_g)]

"""
DTOs
"""

args, serv_times, wait_times, serv_energies = [], [], [], []
th_cs = np.linspace(th_c_min, th_c_max, th_c_num, dtype=np.float64)


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
    trajectory: Dict
    cost_estimate: Tuple
    current_position: tf.Variable
    reporting_period: float = r_p
    timestamp: datetime = datetime.utcfromtimestamp(ntp_client.request(ntp_url).tx_time)

    def __post_init__(self):
        assert self.seq_num >= 0
        assert self.node_id in server_node_ids
        assert self.node_type in ['BS', 'UAV']
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

        self.trajectory = trajectory_  # The trajectory will be empty for the 'BS' and for a 'UAV' in 'WAIT'
        self.current_position = current_position_  # The current position will be [0.0, 0.0, h_bs] for the 'BS'

        self.users = []
        self.env_nodes = {_node: None for _node in server_node_ids}

    def register(self):
        with self.lock:
            if not self.mw_registered:
                middleware.register(self)
                self.mw_registered = True

    def send(self, frame_type, gn_id=None):
        with self.lock:
            if frame_type == 'CTL':
                self.cf_seq_num += 1
                middleware.update(self, ControlFrame([gn_id], self.cf_seq_num, self.id,
                                                     self.type, self.state_flag, frame_type,
                                                     self.trajectory, (), self.current_position))
            elif frame_type == 'REQ' and gn_id is not None and gn_id in ground_node_ids:
                return self.cost_computation_with_piggybacking(gn_id)
            else:
                raise NotImplementedError(f'[ERROR] {self.id} ServiceNode send: Unknown frame_type {frame_type}.')

    def receive(self, src_id, cf):
        frame_type = cf.frame_type

        if frame_type == 'CTL':
            with self.lock:
                self.env_nodes[src_id] = cf
        elif frame_type == 'REQ':
            raise NotImplementedError(f'[INFO] {self.id} ServiceNode receive: REQ receive handled in MoM.')
        else:
            raise NotImplementedError(f'[ERROR] {self.id} ServiceNode receive: Unknown frame_type {frame_type}.')

    def spread_maximization(self):
        r_nxt, th_nxt = 0.0, 0.0

        if self.type == 'UAV':
            f, x = self.state_flag, self.current_position
            r_u, th_u = tf.norm(x), tf.math.atan(x[1] / x[0])
            i_ = np.random.choice([_ for _ in range(th_c_num)])
            i = min([_ for _ in range(n_l)], key=lambda _x: abs(r_u - r_us[_x]))
            peers = {_k: _v.current_position for _k, _v in self.env_nodes.items() if _v.state_flag == 'WAIT'}

            v_r = o_star[tf.argmin(s_wait - r_u)]

            def vel_fn(th_c):
                return (v_r ** 2 + (r_u * th_c) ** 2) ** 0.5

            def obj_fn(th_c):
                return nu * (mobility_pwr(vel_fn(th_c)) - p_avg) * d_01

            constraints = ({'type': 'ineq', 'fun': lambda th_c: v_max - vel_fn(th_c)})
            th_c_star = abs(minimize(obj_fn, th_cs[i_], method='SLSQP', constraints=constraints, tol=d_c).x[0])

            if v_r > 0.0:
                r_nxt, th_nxt = r_us[i + 1], th_u + th_c_star * d_00
            elif v_r < 0.0:
                r_nxt, th_nxt = r_us[i - 1], th_u - th_c_star * d_00
            else:
                min_peer = min(peers.values(), key=lambda _o: abs(th_u - tf.math.atan(_o[1] / _o[0])))

                th_peer = tf.math.atan(min_peer[1] / min_peer[0])
                sgn = min([1, -1], key=lambda _sgn: abs((th_u + _sgn * th_c_star * d_00) - th_peer))

                r_nxt, th_nxt = r_us[i], th_u + th_u + sgn * th_c_star * d_00

        self.trajectory['v'] = tf.constant([], dtype=tf.float64)
        self.trajectory['p'] = tf.constant([[]], dtype=tf.float64)
        self.current_position = tf.Variable([r_nxt * np.cos(th_nxt), r_nxt * np.sin(th_nxt)], dtype=tf.float64)

    @staticmethod
    def frequency_reuse_1(x_txs, x_rx):
        avg_ints = []
        n_p = len(x_txs)

        for x_tx in x_txs:
            xy_dists = tf.norm(tf.subtract(x_tx[:, :2], x_rx[:, :2]))
            hts = tf.constant(abs(x_tx[0, 2] - x_rx[0, 2]), shape=xy_dists.shape, dtype=tf.float64)

            min_d = np.min(tf.sqrt(tf.add(tf.square(xy_dists), tf.square(hts))).numpy(), axis=0)
            avg_int = (min_d ** ap) + (kp * (min_d ** ap_))

            if avg_int > ((1 / snr_deg_tol) - 1) * (1 / (n_p * snr_0)):
                return 0.0, False

        return np.mean(avg_ints), True

    @staticmethod
    def frequency_reuse_2(x_tx, x_rxs):
        n_p = len(x_rxs)
        for x_rx in x_rxs:
            xy_dists = tf.norm(tf.subtract(x_tx[:, :2], x_rx[:, :2]))
            hts = tf.constant(abs(x_tx[0, 2] - x_rx[0, 2]), shape=xy_dists.shape, dtype=tf.float64)

            min_d = np.min(tf.sqrt(tf.add(tf.square(xy_dists), tf.square(hts))).numpy(), axis=0)
            avg_int = (min_d ** ap) + (kp * (min_d ** ap_))

            if avg_int > ((1 / snr_deg_tol) - 1) * (1 / (n_p * snr_0)):
                return 0.0, False

        return True

    def frequency_reuse(self, reqs, gn_id):
        x = self.current_position
        traj = self.trajectory['p']
        tx_rx_pairs = [tx_rx_map[_req] for _req in reqs
                       if tx_rx_map[_req][0] != self.id or tx_rx_map[_req][1] != self.id]

        x_tx = tf.concat([ground_node_map[gn_id], h_gn], axis=1)
        x_rx = tf.concat([tf.concat([x, traj], axis=0), h_u], axis=0)

        x_txs, x_rxs = [], []
        for tx, rx in tx_rx_pairs:
            tx_node, rx_node = self.env_nodes[tx], self.env_nodes[rx]
            x_tx, x_tx_traj = tx_node.current_position, tx_node.trajectory['p']
            x_rx, x_rx_traj = tx_node.current_position, tx_node.trajectory['p']

            h_tx = h_u if tx_node.type == 'UAV' else h_gn
            h_rx = h_u if rx_node.type == 'UAV' else h_bs

            x_txs.append(tf.concat([tf.concat([x_tx, x_tx_traj], axis=0), h_tx], axis=1))
            x_rxs.append(tf.concat([tf.concat([x_rx, x_rx_traj], axis=0), h_rx], axis=1))

        # Analyze reuse at counterpart RXs due to this TX
        reuse = self.frequency_reuse_2(x_tx, x_rxs)

        # Analyze reuse at this RX due to counterpart TXs
        avg_int, reuse = self.frequency_reuse_1(x_txs, x_rx) if reuse else 0.0, False

        return avg_int, reuse

    def channel_wait(self, gn_id):
        with lock:
            ch_waits, reuse_map = {}, {}
            ch_map = {_k: _v for _k, _v in channel_map}

        for chn, chn_els in ch_map.items():
            ch_wait, avg_int, reuse = 0.0, 0.0, 0

            if len(chn_els['serv']) > 0:
                ch_wait = max([requests[_el]['serv_time'] for _el in chn_els['serv']])

                if len(chn_els['wait']) == 0:
                    avg_int, reuse = self.frequency_reuse(chn_els['serv'], gn_id)
                else:
                    i, j, n_wait = 0, 0, len(chn_els['wait'])

                    while i < n_wait:
                        ch_state, fr_peers = '0', [chn_els['wait'][i]]
                        for j in range(i + 1, n_wait):
                            ch_state_ = str(requests[chn_els['wait'][j]]['reuse'])
                            if ch_state_ == '0':
                                break
                            else:
                                fr_peers.append(chn_els['wait'][j])
                                ch_state = ''.join([ch_state, ch_state_])
                        i = j
                        if i == n_wait - 1 and ch_state[-1]:
                            avg_int, reuse = self.frequency_reuse(fr_peers, gn_id)
                            if reuse:
                                continue
                        ch_wait += max([requests[_fr_peer] for _fr_peer in fr_peers])

            ch_waits[chn] = ch_wait
            reuse_map[chn] = (avg_int, reuse)

        chn = min([_ for _ in range(n_c)], key=lambda _idx: ch_waits[ch_map[_idx]])
        return chn, ch_waits[chn], reuse_map[chn]

    def model(self, ch_w_time):
        if self.type == 'UAV' and ch_w_time > 0.0:
            t_elapsed = 0.0

            if self.state_flag == 'COMM':
                wps, vels = self.trajectory['p'], self.trajectory['v']
                ts = tf.divide(tf.norm(tf.roll(wps, shift=-1, axis=0)[:-1, :] - wps[:-1, :], axis=1),
                               tf.where(tf.equal(vels[:-1], 0.0), tf.ones_like(vels[:-1]), vels[:-1]))

                for idx in range(ts.shape[0]):
                    t_elapsed += ts[idx].numpy()

                    if t_elapsed >= ch_w_time:
                        self.current_position = wps[idx]
                        self.trajectory['p'] = wps[idx:]
                        self.trajectory['v'] = vels[:idx]
                        return

            self.state_flag = 'WAIT'
            self.spread_maximization()

        return self.current_position, self.trajectory['p'], self.trajectory['v']

    @staticmethod
    def direct(x_gn, avg_int):
        link_perf = LinkPerformance(avg_int)

        d_gb = np.sqrt(np.add(np.square(h_bs), np.square(np.linalg.norm(x_gn.numpy()))))
        phi_gb = np.arcsin(np.divide(h_bs, d_gb))

        r_los_gb, r_nlos_gb = tf.Variable(0.0, dtype=tf.float64), tf.Variable(0.0, dtype=tf.float64)

        link_perf.ra_tgpt(d_gb, phi_gb, r_los_gb, r_nlos_gb, n_w)

        phi_degrees_gb = (180.0 / pi) * phi_gb

        p_los_gb = 1 / (1 + (z_1 * tf.exp(-z_2 * (phi_degrees_gb - z_1))))
        p_nlos_gb = 1 - p_los_gb

        r_bar_gb = np.add(np.multiply(p_los_gb, r_los_gb.numpy()), np.multiply(p_nlos_gb, r_nlos_gb.numpy()))

        return data_len / r_bar_gb, 0.0

    @staticmethod
    def decode(x_gn, avg_int, d_p, d_v, d_lp):
        link_perf = LinkPerformance(avg_int)

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

    @staticmethod
    def forward(avg_int, f_p, f_v, f_lp):
        link_perf = LinkPerformance(avg_int)

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

    def service(self, x_gn, avg_int, *args_):
        if self.type == 'UAV':
            d_p, d_v, d_lp, f_p, f_v, f_lp = args_
            t_ub, t_p_ub, e_ub, e_p_ub = self.forward(avg_int, f_p, f_v, f_lp)
            t_gu, t_p_gu, e_gu, e_p_gu = self.decode(x_gn, avg_int, d_p, d_v, d_lp)
            serv_time, energy, end_position = t_gu + t_p_gu + t_ub + t_p_ub, e_gu + e_p_gu + e_ub + e_p_ub, f_p[-1]
        else:
            serv_time, energy = self.direct(x_gn, avg_int)
        return serv_time, energy

    def cost_computation_with_piggybacking(self, gn_id):
        x_gn = ground_node_map[gn_id]

        ch, ch_w_time, reuse_with_int = self.channel_wait(gn_id)
        avg_int, reuse = reuse_with_int

        if self.type == 'UAV':
            x_u, _wp, _vel = self.model(ch_w_time)

            st = tf.argmin(tf.reduce_sum(tf.norm(s_comm - tf.concat(x_u, x_gn, axis=1), axis=1), axis=1))

            wp_, vel_ = p_star[st], v_star[st]
            wp, vel = tf.concat([_wp, wp_], axis=0), tf.concat([_vel, vel_], axis=0)

            mp = int(wp.shape[0] / 2)
            s_time, s_nrg = self.service(x_gn, avg_int, wp[:mp], vel[:mp], 0.0, wp[mp:], vel[mp:], 0.0)
        else:
            s_time, s_nrg = self.service(x_gn, avg_int)

        return s_time, ch_w_time, s_nrg, ch, reuse, avg_int

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
        with self.lock:
            if self.running and node.id in server_node_ids:
                self.nodes[node.id] = node
                return True
        return False

    def update(self, src_node, cf):
        k = src_node.id

        with self.lock:
            if self.running:
                dst_nodes = [_v for _k, _v in self.nodes.items() if _k != k]
                self.notify(dst_nodes, k, cf)

    def notify(self, dst_nodes, src_id, cf):
        with self.lock:
            if self.running:
                [node.receive(src_id, cf) for node in dst_nodes]

    @staticmethod
    def conflict_resolution(gn_id):
        costs = []
        for server_node in server_nodes:
            costs.append(server_node.send('REQ', gn_id))

        nd = min([_ for _ in range(len(costs))], key=lambda _idx: ((1 - (nu * p_avg)) * costs[_idx][0]) +
                                                                  costs[_idx][1] + (nu * costs[_idx][2]))

        return server_nodes[nd], costs[nd]

    def expunge(self, node):
        with self.lock:
            if self.running and node.id in server_node_ids:
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
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + (p3 * (v ** 3)) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5)


def gn_request(req_id):
    """
    Simpy queueing model: GN request
    """
    ground_node_id = ground_node_ids[np.random.choice(gn_indices)]

    s_node, (s_time, w_time, s_energy, chn_id, reused, avg_intr) = middleware.conflict_resolution(ground_node_id)

    channel_map[chn_id]['wait'].append(req_id)
    requests[req_id] = {'wait_time': w_time, 'serv_time': s_time,
                        'serv_energy': s_energy, 'channel': chn_id, 'reuse': reused, 'avg_int': avg_intr}

    if s_node.type == 'UAV':
        tx_rx_map[req_id] = (ground_node_id, s_node.id)
        tx_rx_map[req_id] = (s_node.id, server_nodes[0].id)
    else:
        tx_rx_map[req_id] = (ground_node_id, server_nodes[0].id)

    with channel_resources[chn_id].request() as req:
        yield req
        channel_map[chn_id]['serv'] = req_id
        channel_map[chn_id]['wait'].remove(req_id)
        yield env.timeout(s_time)

    channel_map[chn_id]['serv'].remove(req_id)

    wait_times.append(w_time)
    serv_times.append(s_time)
    serv_energies.append(s_energy)


def arrivals():
    """
    Simpy queueing model: Poisson arrivals
    """
    for num in range(n_r):
        req_id = uuid.uuid4()
        env.process(gn_request(req_id))
        yield env.timeout(-np.log(np.random.random_sample()) / id)


@tf.function(experimental_relax_shapes=True)
@tf.autograph.experimental.do_not_convert
def pwr_cost(v):
    return tf.map_fn(mobility_pwr, v, parallel_iterations=n_w)


class LinkPerformance(object):
    """
    Link performance
    """

    def __init__(self, avg_int):
        self.s_0 = snr_0 / avg_int

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

    def u(self, gm, d, los):
        return ((2.0 ** (gm / bw_)) - 1.0) / (self.s_0 * 1.0 if los else kp * (d ** -ap if los else -ap_))

    def los_tgpt(self, d, phi, r_los):
        k = k_1 * np.exp(k_2 * phi)
        ppf_pt = rice.ppf(0.9999999999, k) ** 2.0
        df, nc, y = 2, (2 * k), (k + 1.0) * (1.0 / (self.s_0 * (d ** -ap)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * self.s_0 * (d ** -ap))))

        tf.compat.v1.assign(r_los, self.f_z(z_star), use_locking=True)

    def nlos_tgpt(self, d, r_nlos):
        ppf_pt = rayleigh.ppf(0.9999999999) ** 2.0
        df, nc, y = 2, 0, 1.0 / (snr_0 * (kp * (d ** -ap_)))

        z_star = self.bisect(self.f, df, nc, y, 0.0, self.z(bw_ * np.log2(1 + ppf_pt * self.s_0 * kp * (d ** -ap_))))

        tf.compat.v1.assign(r_nlos, self.f_z(z_star), use_locking=True)

    def ra_tgpt(self, d, phi, r_los, r_nlos, num_workers):
        with ThreadPoolExecutor(num_workers) as executor:
            executor.submit(self.nlos_tgpt, d, r_nlos)
            executor.submit(self.los_tgpt, d, phi, r_los)


"""
Core operations
"""

min_pwr = mobility_pwr(22.0)
env, lock = Environment(), Lock()
middleware = MessagingMiddleware()
channel_ids = [uuid.uuid4() for _ in range(n_c)]
ground_node_ids = [uuid.uuid4() for _ in range(n_g)]
server_node_ids = [uuid.uuid4() for _ in range(n_u + 1)]
channel_resources = {ch_id: Resource(env) for ch_id in channel_ids}
uavs = [tf.Variable([u0], dtype=tf.float64), tf.Variable([u1], dtype=tf.float64), tf.Variable([u2], dtype=tf.float64)]

server_nodes = [ServiceNode(server_node_ids[0], 'BS', x_bs, None)]
[server_nodes.append(ServiceNode(server_node_ids[_u], 'UAV', uavs[_u],
                                 {'p': tf.Variable([[]], dtype=tf.float64),
                                  'v': tf.Variable([], dtype=tf.float64)})) for _u in range(n_u)]

requests, tx_rx_map = {}, {}
server_node_map = {server_node.id for server_node in server_nodes}
ground_node_map = {ground_node_ids[_i]: x_gns[_i] for _i in range(n_g)}
channel_map = {ch_id: {'wait': [], 'serv': []} for ch_id in channel_ids}

try:

    with open(policy_file, 'r') as file:
        for line in file.readlines():
            # noinspection RegExpUnnecessaryNonCapturingGroup
            args.append(tf.strings.to_number(re.findall(r'[-+]?(?:\d*\.*\d+)', line.strip()), tf.float64))

except Exception as e:
    print(f'[ERROR] MAESTRO-X: Exception caught while parsing {policy_file}: {tb.print_tb(e.__traceback__)}.')

uid, nu, p_av, ell, bs_delta_s, uav_delta_s, delta_s, bs_energy_s = args[:8]
s_wait, s_comm, uav_energy_s, a_wait, a_comm, energy_s = args[8:14]
p_star, v_star, xi_star, o_star, u_star = args[14:]

assert uid == agent_id and data_len == ell and p_avg == p_av

env.process(arrivals())
env.run()

total_time = np.mean(np.add(wait_times, serv_times))
p_avg = np.mean(np.divide(serv_energies, total_time))

print(f'[DEBUG] MAESTRO-X: Payload Size = {data_len / 1e6} Mb | Average Comm Delay = {np.mean(serv_times)} seconds.')

print(f'[DEBUG] MAESTRO-X: Payload Size = {data_len / 1e6} Mb | Average Wait Delay = {np.mean(wait_times)} seconds.')

print(f'[DEBUG] MAESTRO-X evaluate: {n_u} UAV-relays | M/G/{n_c} queuing at the data channels | '
      f'Payload Size = {data_len / 1e6} Mb | UAV Power Consumption = {p_avg / 1e3} kW | '
      f'Average Total Service Delay (Wait + Comm) = {total_time} seconds.')
