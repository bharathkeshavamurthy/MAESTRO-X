"""
This script constitutes a "pilot" evaluation of the CVXPy module: a necessary utility for CSCA-ADMM non-adaptive SoA &
SCA convergence evaluations -- particularly, the Consensus Optimization submodule with ADMM.

Simplified Subset (ADMM) Debugging | Superset: src/uav-mobility/model-evaluations/ADMMEvaluation.py

Reference: Consensus Optimization, CVXPy Documentation <https://www.cvxpy.org/examples/applications/consensus_opt.html>

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

# The imports
import numpy as np
from cvxpy import *
from numpy.linalg import norm
from collections import namedtuple
from multiprocessing import Pipe, Process
from numpy.random import uniform, random_sample

"""
Global Definitions
"""

pri_s = namedtuple('vars', ['idx', 'zs', 'vs', 'r_u_s'])
lce_s = namedtuple('lce_s', ['z_hats', 'z_dots', 'z_tildes', 'z_bars', 'v_tildes', 'v_bars', 'r_u_hats', 'r_u_tildes'])

"""
Configurations
"""

pi = np.pi
np.random.seed(6)
utip, v0, p1, p2, p3 = 200.0, 7.2, 580.65, 790.6715, 0.0073
rho, sigma_mul, sigma_alpha, tmax_csca, lmax_admm = 3.5, 0.25, 0.30, 10000, 10000
v_max, p_avg = 55.0, np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[0]
data_len, arr_rates = [1e6, 10e6, 100e6][0], {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.56e-4}
radius, num_slots, num_uavs, num_gns, bs_ht, uav_ht, gn_ht = 1e3, 10000, 3, 10000, 80.0, 200.0, 0.0
snr_0, bw, alpha_los, alpha_nlos, beta_los, beta_nlos, z1, z2 = 1e4, 5e6, 2.0, 2.8, 1e-4, 1e-3, 9.61, 0.16

"""
Node(s) Deployments (BS, GNs, and UAVs)
"""

z_bs = np.array([0.0, 0.0, bs_ht])

v_uavs, r_relays = np.zeros((num_slots, num_uavs)), np.zeros((num_slots, num_gns))

g_rs, g_ths = uniform(0, radius ** 2, num_gns) ** 0.5, uniform(0, 2 * pi, num_gns)
z_gns = np.array(list(zip(g_rs * np.cos(g_ths), g_rs * np.sin(g_ths), np.repeat(gn_ht, num_slots))))

u_r, u_ths = uniform(0, radius ** 2) ** 0.5, uniform(0, 2 * pi, num_slots)
z_uavs = np.repeat(list(zip(u_r * np.cos(u_ths), u_r * np.sin(u_ths), np.repeat(uav_ht, num_slots))), num_uavs, axis=0)

"""
UAV Power Consumption Evaluation Routine
"""


def mobility_power(v):
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
           (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


"""
Channel Model Routines
"""


def d(vec_1, vec_2): return norm(vec_1[:1] - vec_2[:1])
def phi(vec_1, vec_2): return np.arctan(abs(vec_1[2] - vec_2[2]) / d(vec_1, vec_2))
def plos(vec_1, vec_2): return 1 / (1 + z1 * np.exp(-z2 * (phi(vec_1, vec_2) - z1)))
def params(vec_1, vec_2): return (alpha_los, beta_los) if plos(vec_1, vec_2) > 0.5 else (alpha_nlos, beta_nlos)


def throughput(tx, rx):
    return bw * np.log2(1 + snr_0 * gain(tx, rx))


def gain(tx, rx):
    a, b = params(tx, rx)
    return (10 ** (0.1 * b)) * (d(tx, rx) ** -a)


def gain_loc_d(z_uav_t, z_gn):
    d_t = d(z_uav_t, z_gn)
    a_t, b_t = params(z_uav_t, z_gn)
    return np.transpose((((-10 ** (0.1 * b_t)) * a_t) / (d_t ** (a_t + 2))) * (z_uav_t - z_gn))


def gain_aprx(z_uav, z_uav_t, z_gn):
    a_t, b_t = params(z_uav_t, z_gn)
    d_, d_t = d(z_uav, z_gn), d(z_uav_t, z_gn)
    return (((0.5 * a_t + 1) * (d_t ** 2)) - (0.5 * a_t * (d_ ** 2))) / ((10 ** (-0.1 * b_t)) * (d_t ** (a_t + 2)))


"""
Constraints
"""


def avg_pwr(y, pwr_avg):
    k, n = num_gns, num_slots
    return True if np.mean([y[_k][1][_i] for _k in range(k) for _i in range(n)]) <= pwr_avg else False


def uav_vel(y, vel_max):
    k, n = num_gns, num_slots
    return True if sum([0 if y[_k][1][_i] <= vel_max else 1 for _k in range(k) for _i in range(n)]) == 0 else False


"""
Consensus Optimization Routines | Subset of the CSCA-ADMM framework evaluated in NonAdaptiveSoAEvaluation
"""


def surrogate_1(_m, tau, z_uavs_, z_gns_, z_uavs_t_):
    m, d_loc_diff = num_uavs, tau * d(z_uavs_[_m], z_uavs_t_[_m]) ** 2
    gs = {_n: snr_0 * gain(z_uav_t_, z_gns_) for _n, z_uav_t_ in enumerate(z_uavs_t_)}
    gs_loc_dd = {_n: snr_0 * -gain_loc_d(z_uavs_t_[_n], z_gns_) * (z_uavs_[_n] - z_uavs_t_[_n]) for _n in range(m)}

    c1, c1_ = 1 + sum(gs.values()), 1 + sum([_g for _n, _g in gs.items() if _n != _m])
    c2, c2_ = sum(gs_loc_dd.values()), sum([_g for _n, _g in gs_loc_dd.items() if _n != _m])
    u1 = sum([snr_0 * gain_loc_d(z_uavs_t_[_n], z_gns_) * (z_uavs_[_n] - z_uavs_t_[_n]) for _n in range(m) if _n != _m])
    u2 = sum([snr_0 * (gain_aprx(z_uavs_[_n], z_gns_, z_uavs_t_[_n]) - gain(z_uavs_t_[_n], z_gns_)) for _n in range(m)])

    return (bw * np.log2(c1 / c1_)) - d_loc_diff + (bw * ((u2 / c1) + (c2 / c1) - (u1 / c1_) - (c2_ / c1_)))


def surrogate_2(tau, z_uav_, z_uav_t_):
    m, d_loc_diff = num_uavs, tau * d(z_uav_, z_uav_t_) ** 2
    ph = ((snr_0 * (gain_aprx(z_uav_, z_uav_t_, z_bs) - gain(z_uav_t_, z_bs))) / (1 + snr_0 * gain(z_uav_t_, z_bs))) - \
        d_loc_diff + ((snr_0 * -gain_loc_d(z_uav_t_, z_bs) * (z_uav_ - z_uav_t_)) / (1 + snr_0 * gain(z_uav_t_, z_bs)))
    return bw * ph + throughput(z_uav_t_, z_bs)


def work(i, tau_t, zs_t, fn, pipe):
    m, k = num_uavs, num_gns
    vs = Variable((m,), value=v_uavs[i, :])
    zs = Variable((m, 3), value=z_uavs[:, i, :])
    r_u_s = Variable((k,), value=r_relays[i, :])

    lbd6s = Parameter((k,), value=np.zeros((k,)))
    lbd7s = Parameter((k,), value=np.zeros((k,)))
    lbd8s = Parameter((m,), value=np.zeros((m,)))
    lbd9s = Parameter((m,), value=np.zeros((m,)))
    lbd1s = Parameter((m, 3), value=np.zeros((m, 3)))
    lbd2s = Parameter((m, 3), value=np.zeros((m, 3)))
    lbd3s = Parameter((m, 3), value=np.zeros((m, 3)))
    lbd4s = Parameter((k, m, 3), value=np.zeros((k, m, 3)))

    v_bars = Parameter((m,), value=np.zeros((m,)))
    r_u_hats = Parameter((k,), value=np.zeros(k, ))
    v_tildes = Parameter((m,), value=np.zeros((m,)))
    r_u_tildes = Parameter((k,), value=np.zeros(k, ))
    z_hats = Parameter((m, 3), value=np.zeros((m, 3)))
    z_dots = Parameter((m, 3), value=np.zeros((m, 3)))
    z_bars = Parameter((m, 3), value=np.zeros((m, 3)))
    z_tildes = Parameter((m, 3), value=np.zeros((m, 3)))

    v_max_, p_avg_ = Parameter((1,), value=v_max), Parameter((1,), value=p_avg)
    surr_2_ = Parameter((1,), value=np.sum([surrogate_2(tau_t, zs[_m], zs_t[_m]) for _m in range(m)]))
    surr_1_ = Parameter((k,), value=np.sum([surrogate_1(_m, tau_t, zs, z_gns, zs_t) for _m in range(m)]))
    constraints_ = [vs <= v_max_, r_u_hats <= surr_1_, np.sum(r_u_hats) <= surr_2_, mobility_power(vs) <= p_avg_]

    fn -= (rho / 2) * (sum_squares(zs - z_tildes + lbd1s) + sum_squares(zs - z_bars + lbd2s) +
                       sum_squares(v_tildes - vs + lbd8s) + sum_squares(v_bars - vs + lbd9s) +
                       sum_squares(z_hats - z_tildes + lbd3s) + sum_squares(z_dots - z_tildes + lbd4s) +
                       sum_squares(r_u_hats - r_u_tildes + lbd6s) + sum_squares(r_u_s - r_u_tildes + lbd7s))  # Core

    prob = Problem(objective=Maximize(fn), constraints=constraints_)
    while True:
        prob.solve()
        pipe.send(pri_s(i, zs.value, vs.value, r_u_s.value))
        z_hats, z_dots, z_tildes, z_bars, v_tildes, v_bars, r_u_hats, r_u_tildes = pipe.recv()

        lbd2s.value += zs - z_bars
        lbd9s.value += v_bars - vs
        lbd1s.value += zs - z_tildes
        lbd8s.value += v_tildes - vs
        lbd3s.value += z_hats - z_tildes
        lbd4s.value += z_dots - z_tildes
        lbd7s.value += r_u_s - r_u_tildes
        lbd6s.value += r_u_hats - r_u_tildes  # Update the lambdas after sending the primals and receiving the LCE vars


def obj_fn(y): return min([throughput(z_bs, z_gns[_k]) + y[_k][1] for _k in range(num_gns)])


def admm(tau_t, zs_t):
    pipes, procs = [], []
    for i in range(num_slots):
        loc, rem = Pipe()
        pipes.append(loc)
        procs.append(Process(target=work, args=(i, tau_t, zs_t, obj_fn, rem)))
        procs[-1].start()

    zs, vs, r_u_s = z_uavs, v_uavs, r_relays
    for l_idx in range(lmax_admm):
        prxs = [pipe.recv() for pipe in pipes]
        prxs.sort(key=lambda pipe: pipe.idx, reverse=False)
        zs, vs, r_u_s = [prx.z_uavs for prx in prxs], [prx.v_uavs for prx in prxs], [prx.r_relays for prx in prxs]
        ptx_zs, ptx_vs, ptx_r_u_s = np.mean(zs), np.mean(vs), np.mean(r_u_s)
        [pipe.send(lce_s(ptx_zs, ptx_zs, ptx_zs, ptx_zs, ptx_vs, ptx_vs, ptx_r_u_s, ptx_r_u_s)) for pipe in pipes]

    [proc.terminate() for proc in procs]
    return np.array(zs), np.array(vs), np.array(r_u_s)


# Run Trigger
if __name__ == '__main__':
    admm(random_sample(), z_uavs)
