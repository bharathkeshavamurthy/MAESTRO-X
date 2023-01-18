"""
This script constitutes functional and performance evaluations of the SMDP-HCSO augmented policy for a circular cell of
radius $a$ meters consisting of a system of uniformly-distributed Ground Nodes (GNs), a cellular Base Station (BS) at
the center of the cell, and a fleet of Unmanned Aerial Vehicles (UAVs) serving as BS relays.

Final Policy Augmentation: JFQ at the BS | PHY & DLL prescient scheduler between the BS and the UAV fleet...

Author: Bharath Keshavamurthy <bkeshav1@asu.edu | bkeshava@purdue.edu>
Organization: School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.
              School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2022. All Rights Reserved.
"""

"""
TESTING WITH HEURISTICS AND HARD-CODED VALUES
"""

# The imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import itertools
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from simpy import Environment, Resource


def mobility_pwr(v):
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


def wait(x_u, z_u):
    r_u = tf.norm(x_u, axis=1)
    t_u = time.monotonic() - z_u
    v_r = aw_opt[tf.argmin(tf.abs(s_wait - r_u))]
    idx_choice = np.random.choice([_ for _ in range(100)])
    theta_c = np.linspace(0.0, v_max, 100, dtype=np.float64)[idx_choice]

    def vel_fn(th_c):
        return (v_r ** 2 + (r_u * th_c) ** 2) ** 0.5

    def obj_fn(th_c):
        return (0.99 / p_avg) * (mobility_pwr(vel_fn(th_c)) - p_avg) * (-np.log(0.93) / lambda_arr)

    constraints = ({'type': 'ineq', 'fun': lambda th_c: v_max - vel_fn(th_c)})
    theta_c_star = abs(minimize(obj_fn, theta_c, method='SLSQP', constraints=constraints, tol=d_c).x[0])
    d_u = tf.constant(vel_fn(theta_c_star) * t_u, dtype=tf.float64)

    if v_r < 0.0:
        s_comm_x = tf.gather(s_comm[:, 0, :], tf.where(tf.less_equal(tf.norm(s_comm[:, 0, :], axis=1), r_u)))
    else:
        s_comm_x = tf.gather(s_comm[:, 0, :], tf.where(tf.greater_equal(tf.norm(s_comm[:, 0, :], axis=1), r_u)))
    return s_comm_x[tf.argmin(tf.subtract(d_u, tf.norm(tf.subtract(x_u, s_comm_x), axis=2)))[0].numpy()]


# noinspection PyTypeChecker
def gn_request(z, x_g, deltas):
    [
        tf.compat.v1.assign(x_u, wait(x_u, z[u]), validate_shape=True,
                            use_locking=True) if z[u] > 0.0 else None for u, x_u in enumerate(uav_coords)
    ]

    t = env.now
    st = [
        tf.argmin(tf.reduce_sum(tf.norm(s_comm - tf.concat([x_u, x_g], axis=0), axis=1), axis=1)) for x_u in uav_coords
    ]

    uav_tmps = [(number_of_bs_channels + u, uav_serv_times[s],
                 uav_serv_times[s] + deltas[u + number_of_bs_channels]) for u, s in enumerate(st)]
    bs_tmps = [(ch, bs_serv_times[st[0]], bs_serv_times[st[0]] + deltas[ch]) for ch in range(number_of_bs_channels)]
    k, mu, de_val = min(bs_tmps + uav_tmps, key=lambda tmp: tmp[2])
    deltas[k] += mu

    with res[k].request() as req:
        yield req
        wait_times.append(env.now - t)
        yield env.timeout(mu)

    deltas[k] -= mu
    final_serv_times.append(mu)
    if k >= number_of_bs_channels:
        z[k - number_of_bs_channels] = time.monotonic()
        tf.compat.v1.assign(uav_coords[k - number_of_bs_channels],
                            tf.divide(tf.add(tf.constant([[0.0, 0.0]], dtype=tf.float64), x_g), 2.0),
                            validate_shape=True, use_locking=True)


def arrivals():
    z = {u: 0.0 for u in range(number_of_uavs)}
    deltas = {k: 0.0 for k in range(number_of_uavs + number_of_bs_channels)}
    for i in range(n_comm):
        env.process(gn_request(z, tf.expand_dims(s_comm[i_s[i], 1], axis=0), deltas))
        yield env.timeout(-np.log(np.random.random_sample()) / lambda_arr)


def distribute_ground_nodes():
    a, lambda_g = 1e3, 40 / (np.pi * (1e3 ** 2))
    number_of_gns = int(lambda_g * np.pi * (a ** 2))
    radii = np.random.uniform(0, a ** 2, number_of_gns) ** 0.5
    angles = np.random.uniform(0, 2 * np.pi, number_of_gns)
    x_coords, y_coords = radii * np.cos(angles), radii * np.sin(angles)
    return list(zip(radii, angles)), list(zip(x_coords, y_coords))


def discretize_uav_positions(discretization_level, granularity_for_visualization):
    a, offset = 1e3, 50.0
    uav_positions_polar = np.linspace(offset, a - offset, discretization_level, dtype=np.float64)
    angles = np.linspace(0, 2 * np.pi, granularity_for_visualization, dtype=np.float64)
    cosines, sines = np.cos(angles), np.sin(angles)
    coords = np.array([r * np.einsum('ji', np.vstack([cosines, sines])) for r in uav_positions_polar])
    x_coords, y_coords = coords[:, :, 0].flatten(), coords[:, :, 1].flatten()
    return uav_positions_polar, list(zip(x_coords, y_coords))


number_of_bs_channels = 10
ell, number_of_uavs = 100e6, 0
x_gns_polar, x_gns_rect = distribute_ground_nodes()
s_wait, x_uavs_rect = discretize_uav_positions(25, 10)
arrival_rates = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.5555e-4}
env, lambda_arr, n_comm = Environment(), arrival_rates[ell], 10000
utip, v0, p1, p2, p3, v_min, v_max, d_c = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0, 1e-5

vrs = {1e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -27.5, -27.5, -27.5, -27.5,
             -27.5, -27.5, -33.3, -33.3, -33.3, -33.3, -33.3, -33.3, -38.0, -38.0, -38.0, -38.0],
       10e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -27.5,
              -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -33.3, -33.3, -33.3, -33.3],
       100e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5,
               -22.5, -22.5, -22.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5]}

bs_serv_times_parent = {1e6: 156.7513637784017, 10e6: 1606.2192153136264, 100e6: 22527.00699511905}

s_comm, aw_opt = tf.constant(list(itertools.product(x_uavs_rect, x_gns_rect))), tf.constant(vrs[ell], dtype=tf.float64)

serv_times_parent = {1e6: {1: [1.149834209325119,
                               1.0950801993572561904761904761905, 1.0403261893893933809523809523809,
                               0.98557217942153057142857142857141, 0.98557217942153057142857142857141,
                               0.98557217942153057142857142857141, 0.98557217942153057142857142857141,
                               0.98557217942153057142857142857141, 0.98557217942153057142857142857141,
                               0.98557217942153057142857142857141, 0.98557217942153057142857142857141],
                           2: [0.06193186680153974269170544415937,
                               0.05898273028718070732543375634226, 0.05603359377282167195916206852514,
                               0.05308445725846263659289038070803, 0.05308445725846263659289038070803,
                               0.05308445725846263659289038070803, 0.05308445725846263659289038070803,
                               0.05308445725846263659289038070803, 0.05308445725846263659289038070803,
                               0.05308445725846263659289038070803, 0.05308445725846263659289038070803],
                           3: [0.0539199919993409171398530377389,
                               0.05135237333270563537128860737038, 0.04878475466607035360272417700186,
                               0.04621713599943507183415974663334, 0.04621713599943507183415974663334,
                               0.04621713599943507183415974663334, 0.04621713599943507183415974663334,
                               0.04621713599943507183415974663334, 0.04621713599943507183415974663334,
                               0.04621713599943507183415974663334, 0.04621713599943507183415974663334]},
                     10e6: {1: [13.889086488877673841584158415842,
                                13.661396546437056237623762376238, 13.433706603996438633663366336634,
                                13.20601666155582102970297029703, 13.092171690335512227722772277228,
                                13.092171690335512227722772277228, 13.092171690335512227722772277228,
                                13.092171690335512227722772277228, 13.092171690335512227722772277228,
                                13.092171690335512227722772277228, 13.092171690335512227722772277228],
                            2: [0.43255466781199894975696058607757,
                                0.42546360768393339320356778958449, 0.41837254755586783665017499309141,
                                0.41128148742780228009678219659834, 0.4077359573637695018200857983518,
                                0.4077359573637695018200857983518, 0.4077359573637695018200857983518,
                                0.4077359573637695018200857983518, 0.4077359573637695018200857983518,
                                0.4077359573637695018200857983518, 0.4077359573637695018200857983518],
                            3: [0.37588954353747944661922035267296,
                                0.36972741987293060323202001902258, 0.36356529620838175984481968537221,
                                0.35740317254383291645761935172183, 0.35432211071155849476401918489664,
                                0.35432211071155849476401918489664, 0.35432211071155849476401918489664,
                                0.35432211071155849476401918489664, 0.35432211071155849476401918489664,
                                0.35432211071155849476401918489664, 0.35432211071155849476401918489664]},
                     100e6: {1: [167.76916352901743181256739535341,
                                 165.01884937280403129104989706893, 162.26853521659063076953239878445,
                                 159.51822106037723024801490049996, 158.14306398227052998725615135772,
                                 158.14306398227052998725615135772, 158.14306398227052998725615135772,
                                 158.14306398227052998725615135772, 158.14306398227052998725615135772,
                                 158.14306398227052998725615135772, 158.14306398227052998725615135772],
                             2: [3.0211190185098217378902440317401,
                                 2.9715924772227754798920433099083, 2.9220659359357292218938425880764,
                                 2.8725393946486829638956418662447, 2.8477761240051598348965415053288,
                                 2.8477761240051598348965415053288, 2.8477761240051598348965415053288,
                                 2.8477761240051598348965415053288, 2.8477761240051598348965415053288,
                                 2.8477761240051598348965415053288, 2.8477761240051598348965415053288],
                             3: [2.6204185813406969605158638400772,
                                 2.5774608996793740595238004984365, 2.534503218018051158531737156796,
                                 2.4915455363567282575396738151553, 2.470066695526066807043642144335,
                                 2.470066695526066807043642144335, 2.470066695526066807043642144335,
                                 2.470066695526066807043642144335, 2.470066695526066807043642144335,
                                 2.470066695526066807043642144335, 2.470066695526066807043642144335]}}

uav_coords = []
for u_idx in range(1, 4):
    number_of_uavs += 1
    res = [Resource(env) for _ in range(number_of_bs_channels + number_of_uavs)]
    uav_coords.append(tf.Variable(tf.expand_dims(x_uavs_rect[np.random.choice([_ for _ in range(25 * 10)])], axis=0)))
    for idx, serv_time in enumerate(serv_times_parent[ell][u_idx]):
        p_avg = 1e3 + (0.1e3 * idx)
        wait_times, final_serv_times = list(), list()
        i_s = tf.random.shuffle([_ for _ in range(n_comm)])
        bs_serv_times = np.abs(np.random.normal(loc=bs_serv_times_parent[ell],
                                                scale=bs_serv_times_parent[ell] / 10.0, size=n_comm))
        uav_serv_times = np.abs(np.random.normal(loc=0.45 * serv_time, scale=(0.45 * serv_time) / 10.0, size=n_comm))
        env.process(arrivals())
        env.run()
        print(f'[INFO] SMDPAugmentedPolicyModelII main: Number of UAVs = {number_of_uavs} | '
              f'Number of orthogonal BS channels = {number_of_bs_channels} | Payload size = {ell / 1e6} Mb | '
              f'Arrival rate = {lambda_arr} req/s | Average power constraint = {p_avg / 1e3} kW | '
              f'{number_of_uavs} UAV-relay(s) average queue wait time = {np.mean(wait_times)} s | '
              f'{number_of_uavs} UAV-relay(s) average service time = {np.mean(final_serv_times)} s | '
              f'{number_of_uavs} UAV-relay(s) total service time = {np.mean(wait_times) + np.mean(final_serv_times)} s')
    print('\n')
# The SMDP-HCSO augmented policy evaluation for multiple UAV-relays ends here...
