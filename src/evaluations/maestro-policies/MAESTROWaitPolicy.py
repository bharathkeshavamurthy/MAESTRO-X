"""
This script constitutes the set of operations necessary to visualize the optimal waiting state policy in our MAESTRO
framework, i.e., the UAV's optimal radial and angular velocity actions corresponding to a specific waiting state and
its subsequent transition to a neighboring waiting state (assuming zero GN requests arrive within that time-frame).

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import plotly
import warnings
import numpy as np
import plotly.graph_objs as go
from numpy.random import choice
from scipy.optimize import minimize

"""
Miscellaneous
"""
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Configurations-I: Plotly API settings
"""
plotly.tools.set_credentials_file(username='bkeshav1', api_key='PUYaTVhV1Ok04I07S4lU')

"""
Configurations-II: Simulation parameters
"""
pi = np.pi
np.random.seed(6)
data_lens = [1e6, 10e6, 100e6]
r_th_num, r_interp_num = 100, int(1e4)
a, a_o, r_num, th_num, d_c = 1e3, 50.0, 25, 25, 1e-10
th_us = np.linspace(0, 315.0, th_num, dtype=np.float64)
r_us = np.linspace(a_o, a - a_o, r_num, dtype=np.float64)
arr_rates = {1e6: 1.67e-2, 10e6: 3.33e-3, 100e6: 5.5555e-4}
utip, v0, p1, p2, p3, v_min, v_max = 200.0, 7.2, 580.65, 790.6715, 0.0073, 0.0, 55.0
p_avgs, th_c_num = np.arange(start=1e3, stop=2.2e3, step=0.2e3, dtype=np.float64), int(1e4)

''' TODO: Read this as a tensor from the policy log file instead of copy-pasting it here directly. '''
data_lens_v_rs = {1e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -27.5, -27.5, -27.5, -27.5,
                        -27.5, -27.5, -33.3, -33.3, -33.3, -33.3, -33.3, -33.3, -38.0, -38.0, -38.0, -38.0],
                  10e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -27.5,
                         -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -33.3, -33.3, -33.3, -33.3],
                  100e6: [7.5, 7.5, 7.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5, -22.5,
                          -22.5, -22.5, -22.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5, -27.5]}

"""
Utilities
"""


def mobility_pwr(v):
    """
    UAV mobility power consumption
    """
    return (p1 * (1 + ((3 * (v ** 2)) / (utip ** 2)))) + \
        (p2 * (((1 + ((v ** 4) / (4 * (v0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v0 ** 2)))) ** 0.5) + (p3 * (v ** 3))


"""
Core operations
"""


def trace(i, r_u, th_u, v_r, nu, p_avg, d_00, d_01, r_us_):
    idx_choice = np.random.choice([_ for _ in range(th_c_num)])
    theta_c = np.linspace(0.0, 2 * pi, th_c_num, dtype=np.float64)[idx_choice]

    def vel_fn(th_c):
        return (v_r ** 2 + (r_u * th_c) ** 2) ** 0.5

    def obj_fn(th_c):
        return nu * (mobility_pwr(vel_fn(th_c)) - p_avg) * d_01

    constraints = ({'type': 'ineq', 'fun': lambda th_c: v_max - vel_fn(th_c)})
    theta_c_star = abs(minimize(obj_fn, theta_c, method='SLSQP', constraints=constraints, tol=d_c).x[0])

    if v_r > 0.0:
        r_nxt, th_nxt, ths, sw = r_us_[i + 1], th_u + ((180 / pi) * theta_c_star * d_00), [], False
    elif v_r < 0.0:
        r_nxt, th_nxt, ths, sw = r_us_[i - 1], th_u - ((180 / pi) * theta_c_star * d_00), [], False
    else:
        r_nxt, th_nxt, ths, sw = r_us_[i], th_u, np.linspace(0, 360.0, r_th_num), True

    print(f'r_u = {r_u} | r_next = {r_nxt} | th_next = {th_nxt} degrees | '
          f'v_r* = {v_r} m/s | theta_c* = {(180 / pi) * theta_c_star} deg/s | v* = {vel_fn(theta_c_star)} m/s.')

    return vel_fn(theta_c_star), r_nxt, th_nxt, ths, sw


def visualize():
    for d_len, d_len_vrs in data_lens_v_rs.items():
        switch, switch_index, mems = 0, 0, list()
        trs, r_us_new = list(), np.linspace(a_o, a - a_o, r_interp_num)

        r_us_interp = np.append(r_us, r_us_new[np.argmin(np.abs(np.interp(r_us_new, r_us, d_len_vrs) - 0.0))])
        r_us_interp = np.sort(r_us_interp)  # Perform the mapping r <=> v_r_opt for the zero radial velocity value

        d_len_vrs.append(0.0)  # Adding the v_r_opt = 0.0 value for "full waiting" | Circle around in this radius level
        d_len_vrs.sort(key=lambda x: x, reverse=True)  # Sort the v_r_opts in descending order for post-processing visl.

        for i in range(r_num + 1):
            mems.append(trace(i, r_us_interp[i], mems[i - 1][2] if i > 0 else th_us[0], d_len_vrs[i],
                              0.99 / p_avgs[1], p_avgs[1], 1.0, -np.log(0.93) / arr_rates[d_len], r_us_interp))

            if len(mems[i][3]) > 0:
                trs.append(go.Scatterpolar(theta=mems[i][3],
                                           r=[r_us_interp[i] for _ in range(r_th_num)],
                                           marker=dict(size=6, symbol='circle'), mode='markers'))
            else:
                trs.append(go.Scatterpolar(r=[r_us_interp[i], mems[i][1]],
                                           theta=[mems[i - 1][2] if i > 0 else th_us[0], mems[i][2]],
                                           name=f'L = {d_len / 1e6} Mb | v = {mems[i][0]}', mode='lines+markers'))

            if mems[i][4]:
                switch, switch_index, mems = 1, i, list()
                break  # The UAV has reached a lvl of v_r = 0: this ends a stage from 0-->r* | Reverse-track from r_last

        if switch:
            for j in range(r_num, switch_index, -1):
                i = r_num - j
                mems.append(trace(j, r_us_interp[j], mems[i - 1][2] if i > 0 else th_us[-1], d_len_vrs[j],
                                  0.99 / p_avgs[1], p_avgs[1], 1.0, -np.log(0.93) / arr_rates[d_len], r_us_interp))

                trs.append(go.Scatterpolar(r=[r_us_interp[j], mems[i][1]],
                                           theta=[mems[i - 1][2], mems[i][2]],
                                           name=f'L = {d_len / 1e6} Mb | v = {mems[i][0]}', mode='lines+markers'))

        plotly.plotly.plot(dict(data=trs, layout=dict(title='Waiting State Optimal Policy Visualization')))


# Run Trigger
if __name__ == '__main__':
    visualize()
