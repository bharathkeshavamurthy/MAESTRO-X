"""
This script visualizes the relationship between the forward velocity and the associated power consumption of a
rotary-wing UAV -- with the mobility/power model extracted from Helicopter Motion Dynamics.

Author: Bharath Keshavamurthy <bkeshava@purdue.edu | bkeshav1@asu.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
              School of Electrical, Computer and Energy Engineering, Arizona State University, Tempe, AZ.

Copyright (c) 2022. All Rights Reserved.
"""

import numpy
import plotly
import plotly.graph_objs as graph_objs

"""
Configurations-I: Plotly API credentials
"""
plotly.tools.set_credentials_file(username='bkeshav1', api_key='BCvYNi3LNNXgfpDGEpo0')

"""
Configurations-II: UAV mobility power consumption model
"""

# The mean rotor-induced velocity ($v_{0}$) in m/s
MEAN_ROTOR_INDUCED_VELOCITY = 7.2

# The maximum UAV velocity ($V_{\text{max}}$) in m/s
MAX_UAV_VELOCITY = 55.0

# The rotor blade tip speed ($U_{\text{tip}}$) in m/s
ROTOR_BLADE_TIP_SPEED = 200.0

# The UAV power consumption profile constant 1 ($P_{1}$) relevant w.r.t blade profile
POWER_PROFILE_CONSTANT_1 = 580.65

# The UAV power consumption profile constant 2 ($P_{2}$) relevant w.r.t induced velocity profile
POWER_PROFILE_CONSTANT_2 = 790.6715

# The UAV power consumption profile constant 3 ($P_{3}$) relevant w.r.t to the parasitic effects
POWER_PROFILE_CONSTANT_3 = 0.0073

# The thrust-to-weight ratio in the rotary-wing UAV motion model ($\kappa_{\text{UAV}}{\triangleq}\frac{T}{W}$
THRUST_TO_WEIGHT_RATIO = 1.0

"""
Core operations
"""


def evaluate_power_consumption(uav_flying_velocity):
    """
    UAV mobility power consumption
    """
    v, u_tip, v_0 = uav_flying_velocity, ROTOR_BLADE_TIP_SPEED, MEAN_ROTOR_INDUCED_VELOCITY
    p_1, p_2, p_3 = POWER_PROFILE_CONSTANT_1, POWER_PROFILE_CONSTANT_2, POWER_PROFILE_CONSTANT_3

    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_3 * (v ** 3)) + \
        (p_2 * (((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) - ((v ** 2) / (2 * (v_0 ** 2)))) ** 0.5)


velocities = numpy.arange(start=0.0, stop=MAX_UAV_VELOCITY, step=1.0, dtype=numpy.float64)

powers = numpy.array([evaluate_power_consumption(_v) for _v in velocities])

plot_trace_mobility = graph_objs.Scatter(x=velocities, y=powers, mode='lines+markers')
plot_trace_communication = graph_objs.Scatter(x=[22.0], y=[90.0], mode='markers')

plot_layout = dict(title='Rotary-Wing UAV Mobility Power Analysis',
                   xaxis=dict(title='Horizontal Flying Velocity in m/s (V)', autorange=True),
                   yaxis=dict(title='UAV Power Consumption in W (P)', type='log', autorange=True))

fig = dict(data=[plot_trace_mobility, plot_trace_communication], layout=plot_layout)
fig_url = plotly.plotly.plot(fig, filename='UAV_Mobility_Power_Analysis', auto_open=False)

print('[INFO] UAVPowerModel main: The plot illustrating the relationship between the '
      f'flying velocity and the associated power consumption of a rotary-wing UAV is given here - {fig_url}.')
