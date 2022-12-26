"""
This paper constitutes the evaluation of the CSCA-ADMM framework from the state-of-the-art -- adapted to our PHY and LL
models. A few of these adaptations include "static bandwidth allocations", "obstacle-free UAV movements", "custom UAV
power & mobility models", "queueing and scheduling in the Link Layer", and "delay analysis".

Reference Paper:

    @ARTICLE{CSCA,
    author={Hu, Qiyu and Cai, Yunlong and Liu, An and Yu, Guanding and Li, Geoffrey Ye},
    journal={IEEE Transactions on Wireless Communications},
    title={Low-Complexity Joint Resource Allocation and Trajectory Design for UAV-Aided Relay Networks With
           the Segmented Ray-Tracing Channel Model},
    year={2020},
    volume={19},
    number={9},
    pages={6179-6195},
    doi={10.1109/TWC.2020.3000864}}

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import warnings
import numpy as np
import cvxpy as cp
from numpy.linalg import norm
from simpy import Environment, Resource
from multiprocessing import Pipe, Process
from numpy.random import choice, uniform, random

"""
Miscellaneous
"""

# Filter user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global utilities for basic operations
decibel, linear = lambda _x: 10.0 * np.log10(_x), lambda _x: 10.0 ** (_x / 10.0)

"""
Configurations-I: Simulation parameters
"""

np.random.seed(6)
pi, num_uavs = np.pi, 3
utip, v0, p1, p2, p3 = 200.0, 7.2, 580.65, 790.6715, 0.0073
depl_env, rf, le_l, le_m, le_h = 'rural', num_uavs, 1, 10, 100
alpha_los, alpha_nlos, beta_los, beta_nlos = 2.0, 2.8, 1e-3, 1e-4
rho, sigma_mul, sigma_alpha, tmax_csca, lmax_admm = 3.0, 0.1, 0.1, 10, 10
v_min, v_max, arr_rates_r = 0.0, 55.0, {1e6: 5 / 60, 10e6: 1 / 60, 100e6: 1 / 360}
radius, num_slots, num_levels, min_dist, bs_ht, uav_ht, gn_ht = 1e3, int(1e4), 25, 25.0, 80.0, 200.0, 0.0
data_len, p_avg = [1e6, 10e6, 100e6][0], np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64)[0]
max_iters, eps_abs, eps_rel, warm_start, verbose, csca_conf, csca_tol = int(1e6), 1e-6, 1e-6, True, True, 5, 1e-5

arr_rates_l = {_k: _v * rf * le_l for _k, _v in arr_rates_r.items()}
arr_rates_m = {_k: _v * rf * le_m for _k, _v in arr_rates_r.items()}
arr_rates_h = {_k: _v * rf * le_h for _k, _v in arr_rates_r.items()}

'''
TODO: Change z1 and z2 according to the deployment environment
TODO: Change bw and n_c according to the deployment environment (Verizon LTE/LTE-A/5G)
'''
if depl_env == 'rural':
    bw, n_c, z1, z2, arr_rates = 10e6, 2, 9.61, 0.16, arr_rates_l
elif depl_env == 'suburban':
    bw, n_c, z1, z2, arr_rates = 20e6, 4, 9.61, 0.16, arr_rates_m
else:
    bw, n_c, z1, z2, arr_rates = 40e6, 8, 9.61, 0.16, arr_rates_h

bw_ = bw / n_c
snr_0 = linear((5e6 * 40) / bw_)

"""
Configurations-II: Deployment settings
"""

''' BS deployment '''
z_bs = np.array([0.0, 0.0, bs_ht])

''' UAV(s) deployment '''

u_ths = uniform(0, 2 * pi, num_slots)
u_r = uniform(0, radius ** 2, num_uavs) ** 0.5

z_uavs = np.reshape(np.array([list(zip(u_r[_m] * np.cos(u_ths),
                                       u_r[_m] * np.sin(u_ths),
                                       np.repeat(uav_ht, num_slots)))
                              for _m in range(num_uavs)]), (num_slots, num_uavs, 3))

''' GNs deployment (uniform-circular) '''

g_rs = np.linspace(start=0.0, stop=radius, num=num_levels)
g_ths = [np.linspace(start=0.0, stop=2 * np.pi, num=int((2 * np.pi * _r) / min_dist) + 1) for _r in g_rs]

z_gns_2d = np.concatenate([_r * np.einsum('ji', np.vstack([np.cos(g_ths[_i]),
                                                           np.sin(g_ths[_i])])) for _i, _r in enumerate(g_rs)], axis=0)

z_gns = np.concatenate([z_gns_2d, np.zeros(shape=(z_gns_2d.shape[0], 1))], axis=1)

num_gns = z_gns.shape[0]
gn_indices = [_ for _ in range(num_gns)]

''' Other initializations (velocities and throughputs) '''
v_uavs, r_relays = choice(np.linspace(v_min, v_max, num_slots), (num_slots, num_uavs)), random((num_slots, num_gns))

"""
Utilities
"""


def mobility_power(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


def gn_request(env, chs, w_times, s_times, serv_times):
    """
    Simpy queueing model: GN request
    """
    arr_time = env.now
    gn_r_idx = np.random.choice(gn_indices)
    k = np.argmin([max([0, len(_k.put_queue) + len(_k.users)]) for _k in chs])

    with chs[k].request() as req:
        yield req
        w_times.append(env.now - arr_time)
        s_times.append(serv_times[gn_r_idx])
        yield env.timeout(serv_times[gn_r_idx])


def arrivals(env, chs, n_r, arr, w_times, s_times, serv_times):
    """
    Simpy queueing model: Poisson arrivals
    """
    for num in range(n_r):
        env.process(gn_request(env, chs,
                               w_times, s_times, serv_times))
        yield env.timeout(-np.log(np.random.random_sample()) / arr)


"""
Core operations
"""


def d(vec_1, vec_2):
    return np.sqrt(np.square(norm(vec_1[:2] - vec_2[:2])) + np.square(vec_1[2] - vec_2[2]))


def phi(vec_1, vec_2):
    return (180 / pi) * np.arcsin(np.abs(vec_1[2] - vec_2[2]) / d(vec_1, vec_2))


def plos(vec_1, vec_2):
    return 1 / (1 + z1 * np.exp(-z2 * (phi(vec_1, vec_2) - z1)))


def params(vec_1, vec_2):
    return (alpha_los, beta_los) if plos(vec_1, vec_2) > 0.5 else (alpha_nlos, beta_nlos)


def gain(tx, rx):
    a, b = params(tx, rx)
    return (10 ** (0.1 * b)) * (d(tx, rx) ** -a)


def throughput(tx, rx):
    return bw_ * np.log2(1 + snr_0 * gain(tx, rx))


def gain_loc_d(z_uav_t, z_gn):
    d_t = d(z_uav_t, z_gn)
    a_t, b_t = params(z_uav_t, z_gn)
    return ((-10 ** (0.1 * b_t)) * a_t * (z_uav_t - z_gn)) / (d_t ** (a_t + 2))


def gain_aprx(z_uav, z_uav_t, z_gn):
    a_t, b_t = params(z_uav_t, z_gn)
    d_, d_t = d(z_uav, z_gn), d(z_uav_t, z_gn)
    return (((0.5 * a_t + 1) * (d_t ** 2)) - (0.5 * a_t * (d_ ** 2))) / ((10 ** (-0.1 * b_t)) * (d_t ** (a_t + 2)))


def uav_vel(y):
    k, n, m = num_gns, num_slots, num_uavs
    return True if np.sum([0 if 0.0 <= y[1][_i][_m] <= v_max else 1
                           for _i in range(n) for _m in range(m)], axis=0) else False


def avg_pwr(y):
    k, n, m = num_gns, num_slots, num_uavs
    return True if np.sum([0 if np.mean([mobility_power(y[1][_i][_m])
                                         for _i in range(n)], axis=0) <= p_avg else 1 for _m in range(m)]) else False


def step_size(y_t_1, y_t, y_t_star, sigma_t):
    k, n, s, a = num_gns, num_slots, sigma_mul, sigma_alpha

    def boundary(yt, ytstar, st):
        return np.min(a * st * np.full((k,), n) * np.mean(ytstar[2] - yt[2], axis=0), axis=0)

    def f_t(y):
        return min([np.sum(y[2][_k], axis=0) + (n * bw_ * np.log2(snr_0 * gain(z_bs, z_gns[_k]))) for _k in range(k)])

    if f_t(y_t_1) - f_t(y_t) < boundary(y_t, y_t_star, sigma_t) or uav_vel(y_t_1) or avg_pwr(y_t_1):
        sigma_t *= s

    return sigma_t


def surrogate_1(_m, tau, z_uavs_, z_gns_, z_uavs_t_):
    m, d_loc_diff = num_uavs, tau * d(z_uavs_[_m], z_uavs_t_[_m]) ** 2

    gs = {_n: np.array([snr_0 * gain(_z_uav_t_, _z_gn_)
                        for _z_gn_ in z_gns_]) for _n, _z_uav_t_ in enumerate(z_uavs_t_)}

    gs_loc_dd = {_n: np.array([snr_0 *
                               np.matmul(np.transpose(-gain_loc_d(z_uavs_t_[_n], _z_gn_)),
                                         (z_uavs_[_n] - z_uavs_t_[_n])) for _z_gn_ in z_gns_]) for _n in range(m)}

    c1 = 1 + np.sum([_g for _g in gs.values()], axis=0)
    c2 = np.sum([_g for _g in gs_loc_dd.values()], axis=0)
    c1_ = 1 + np.sum([_g if _n != _m else 0 for _n, _g in gs.items()], axis=0)
    c2_ = np.sum([_g if _n != _m else 0 for _n, _g in gs_loc_dd.items()], axis=0)

    u1 = np.sum([[snr_0 * np.matmul(np.transpose(gain_loc_d(z_uavs_t_[_n], _z_gn_)),
                                    (z_uavs_[_n] - z_uavs_t_[_n])) for _z_gn_ in z_gns_]
                 if _n != _m else np.zeros(shape=(num_gns,)) for _n in range(m)], axis=0)

    u2 = np.sum([[snr_0 * (gain_aprx(z_uavs_[_n], _z_gn_, z_uavs_t_[_n]) -
                           gain(z_uavs_t_[_n], _z_gn_)) for _z_gn_ in z_gns_] for _n in range(m)], axis=0)

    return (bw_ * np.log2(c1 / c1_)) - d_loc_diff + (bw_ * ((u2 / c1) + (c2 / c1) - (u1 / c1_) - (c2_ / c1_)))


def surrogate_2(tau, z_uav_, z_uav_t_):
    m, d_loc_diff = num_uavs, tau * d(z_uav_, z_uav_t_) ** 2

    ph = ((snr_0 * (gain_aprx(z_uav_, z_uav_t_, z_bs) - gain(z_uav_t_, z_bs))) / (1 + snr_0 * gain(z_uav_t_, z_bs))) - \
         d_loc_diff + ((snr_0 * np.matmul(np.transpose(-gain_loc_d(z_uav_t_, z_bs)), (z_uav_ - z_uav_t_))) /
                       (1 + snr_0 * gain(z_uav_t_, z_bs)))

    return bw_ * ph + throughput(z_uav_t_, z_bs)


def update(y_t, y_t_star, sigma_t):
    return (1 - sigma_t) * y_t + sigma_t * y_t_star


def obj_fn(r_s):
    return min([throughput(z_bs, z_gns[_k]) + r_s[_k] for _k in range(num_gns)])


def csca():
    r_c = obj_fn(r_relays[0, :])
    zs_t, vs_t, r_u_s_t, r_c_t = z_uavs, v_uavs, r_relays, r_c
    tau_t, y_t, sigma_t = uniform(), np.array([z_uavs, v_uavs, r_relays, r_c]), 1.0

    for t_csca in range(tmax_csca):
        y_t_star = admm(tau_t, zs_t,
                        vs_t, r_u_s_t, r_c_t)
        y_t_1 = update(y_t, y_t_star, sigma_t)
        sigma_t = step_size(y_t_1, y_t, y_t_star, sigma_t)
        y_t, (zs_t, vs_t, r_u_s_t, r_c_t) = y_t_1, y_t_1

    return y_t


# noinspection PyTypeChecker
def work(i, fn, tau_t, zs_t, vs_t, r_u_s_t, r_c_t, pipe):
    m, k = num_uavs, num_gns

    r_c = cp.Variable(value=r_c_t)
    vs = cp.Variable((m,), value=vs_t)
    zs = cp.Variable((m, 3), value=zs_t)
    r_u_s = cp.Variable((k,), value=r_u_s_t)
    ys = cp.Variable((m,), value=np.sqrt(np.sqrt(1 + ((vs_t ** 4) /
                                                      (4 * v0 ** 4))) - ((vs_t ** 2) / (2 * v0 ** 2))))

    lbd6s = cp.Parameter((k,), value=random((k,)))
    lbd7s = cp.Parameter((k,), value=random((k,)))
    lbd8s = cp.Parameter((m,), value=random((m,)))
    lbd9s = cp.Parameter((m,), value=random((m,)))
    lbd11s = cp.Parameter((k,), value=random((k,)))
    lbd1s = cp.Parameter((m, 3), value=random((m, 3)))
    lbd2s = cp.Parameter((m, 3), value=random((m, 3)))
    lbd3s = cp.Parameter((m, 3), value=random((m, 3)))
    lbd4s = cp.Parameter((m, 3), value=random((m, 3)))

    v_bars = cp.Parameter((m,), value=vs.value)
    v_tildes = cp.Parameter((m,), value=vs.value)
    z_hats = cp.Parameter((m, 3), value=zs.value)
    z_dots = cp.Parameter((m, 3), value=zs.value)
    z_bars = cp.Parameter((m, 3), value=zs.value)
    z_tildes = cp.Parameter((m, 3), value=zs.value)
    r_u_hats = cp.Parameter((k,), value=r_u_s.value)
    r_u_tildes = cp.Parameter((k,), value=r_u_s.value)
    r_c_tildes = cp.Parameter((k,), value=np.repeat(r_c.value, k))

    surr_2_ = np.sum([surrogate_2(tau_t, zs.value[_m], zs_t[_m]) for _m in range(m)], axis=0)
    surr_1_ = np.sum([surrogate_1(_m, tau_t, zs.value, z_gns, zs_t) for _m in range(m)], axis=0)

    mob_pwr = p1 * (1 + (3 * cp.power(vs, 2)) / (utip ** 2)) + p3 * cp.power(vs, 3) + p2 * ys
    tgpts = (np.array([throughput(_z_gn_, z_bs) for _z_gn_ in z_gns]) + r_u_tildes) * num_slots

    constraints_ = [vs >= 0.0, vs <= v_max, r_c_tildes <= tgpts, r_u_s >= 0.0,
                    r_u_s <= surr_1_, np.sum(r_u_hats, axis=0) <= surr_2_, mob_pwr <= p_avg]

    fn -= (rho / 2) * (cp.sum_squares(zs - z_tildes + lbd1s) +
                       cp.sum_squares(zs - z_bars + lbd2s) + cp.sum_squares(v_tildes - vs + lbd8s) +
                       cp.sum_squares(v_bars - vs + lbd9s) + cp.sum_squares(z_hats - z_tildes + lbd3s) +
                       cp.sum_squares(z_dots - z_tildes + lbd4s) + cp.sum_squares(r_c - r_c_tildes + lbd11s) +
                       cp.sum_squares(r_u_hats - r_u_tildes + lbd6s) + cp.sum_squares(r_u_s - r_u_tildes + lbd7s))

    prob = cp.Problem(objective=cp.Maximize(fn), constraints=constraints_)

    while True:
        prob.solve('SCS', max_iters=max_iters, eps_abs=eps_abs, eps_rel=eps_rel, warm_start=warm_start, verbose=verbose)

        print(f'i = {i} | Primal Value = {prob.value} | Problem Status = {prob.solution.status}.')

        pipe.send((i, zs.value, vs.value, r_u_s.value, r_c.value))
        z_hats.value, z_dots.value, z_tildes.value, z_bars.value, v_tildes.value, \
            v_bars.value, r_u_hats.value, r_u_tildes.value, r_c_tildes.value = pipe.recv()

        lbd2s.value += zs.value - z_bars.value
        lbd9s.value += v_bars.value - vs.value
        lbd1s.value += zs.value - z_tildes.value
        lbd8s.value += v_tildes.value - vs.value
        lbd3s.value += z_hats.value - z_tildes.value
        lbd4s.value += z_dots.value - z_tildes.value
        lbd11s.value += r_c.value - r_c_tildes.value
        lbd7s.value += r_u_s.value - r_u_tildes.value
        lbd6s.value += r_u_hats.value - r_u_tildes.value


def admm(tau_t, zs_t, vs_t, r_u_s_t, r_c_t):
    pipes, procs = [], []
    zs, vs, r_u_s = z_uavs, v_uavs, r_relays

    for i in range(num_slots):
        loc, rem = Pipe()
        pipes.append(loc)

        procs.append(Process(target=work, args=(i, r_c_t, tau_t, zs_t[i, :, :], vs_t[i, :], r_u_s_t[i, :], r_c_t, rem)))

        procs[-1].start()

    for l_idx in range(lmax_admm):
        prxs = [pipe.recv() for pipe in pipes]
        prxs.sort(key=lambda pipe: pipe[0], reverse=False)
        zs = [prx[1] for prx in prxs if prx[1] is not None]
        vs = [prx[2] for prx in prxs if prx[2] is not None]
        r_u_s = [prx[3] for prx in prxs if prx[3] is not None]
        r_c_s_1, r_c_s_2 = [prx[4] for prx in prxs if prx[4] is not None], [obj_fn(_r_u_s) for _r_u_s in r_u_s]

        ptx_vs = np.reshape(np.mean(vs, axis=0), (num_uavs,))
        ptx_zs = np.reshape(np.mean(zs, axis=0), (num_uavs, 3))
        ptx_r_u_s = np.reshape(np.mean(r_u_s, axis=0), (num_gns,))
        ptx_r_c = np.repeat(np.mean(np.concatenate([r_c_s_1, r_c_s_2], axis=0), axis=0), (num_gns,))
        [pipe.send((ptx_zs, ptx_zs, ptx_zs, ptx_zs, ptx_vs, ptx_vs, ptx_r_u_s, ptx_r_u_s, ptx_r_c)) for pipe in pipes]

    [proc.terminate() for proc in procs]
    return np.array([np.array(zs), np.array(vs), np.array(r_u_s), obj_fn(np.mean(r_u_s, axis=0))])


def evaluate():
    z_uavs_opt, v_uavs_opt, r_relays_opt, r_c_opt = csca()
    serv_idxs, w_times, s_times, serv_times = [], [], [], []
    n, m, rate, env = num_slots, num_uavs, arr_rates[data_len], Environment()

    for i in range(n):
        uav_mx = [np.argmin([norm(z_uavs_opt[_i][_m] -
                                  z_gns[i]) for _i in range(n)]) for _m in range(m)]
        mx_uav = np.argmin([norm(z_uavs_opt[uav_mx[_m]][_m] - z_gns[i]) for _m in range(m)])

        i_min, z_s, v_s, r_s = uav_mx[mx_uav], z_uavs_opt[:, mx_uav], v_uavs_opt[:, mx_uav], r_relays_opt[:, i]

        trajs = np.array(z_s[i:i_min + 1] if i < i_min else z_s[i_min:i + 1])
        rates = np.array([throughput(traj_pt, z_gns[i]) for traj_pt in trajs]) + np.array(r_s[i:i_min + 1] if i < i_min
                                                                                          else r_s[i_min:i + 1])

        serv_idxs.append(mx_uav)
        serv_times.append(data_len / np.mean(rates, axis=0))

    env.process(arrivals(env, [Resource(env) for _ in range(n_c)], n, rate, w_times, s_times, serv_times))

    env.run()

    print('[DEBUG] ADMMEvaluation evaluate: '
          f'Payload Size = {data_len / 1e6} Mb | '
          f'Average Wait Delay = {np.mean(w_times)} seconds.')

    print('[DEBUG] ADMMEvaluation evaluate: '
          f'Payload Size = {data_len / 1e6} Mb | '
          f'Average Comm Delay = {np.mean(s_times)} seconds.')

    print('[DEBUG] ADMMEvaluation evaluate: '
          f'{num_uavs} UAV-relays | M/G/{n_c} queuing at the data channels | '
          f'Payload Size = {data_len / 1e6} Mb | UAV Power Consumption = {p_avg / 1e3} kW | '
          f'Average Total Service Delay (Wait + Comm) = {np.mean(np.add(w_times, s_times))} seconds.')


# Run Trigger
if __name__ == '__main__':
    evaluate()
