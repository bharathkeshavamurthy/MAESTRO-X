"""
A script utilizing the TensorFlow Constrained Optimization Library to solve for the optimal angular velocity value,
given the waiting state (UAV position radius level) and waiting action (UAV radial velocity)

Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

import os

# Logging setup for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The imports
import numpy as np
import tensorflow as tf
import tensorflow_constrained_optimization as tfco

"""
Derived from the Convex Optimization Example in the TensorFlow Constrained Optimization Library
Use the following pip install command for tfco: 
"pip install git+https://github.com/google-research/tensorflow_constrained_optimization"
"""


class UAVAngularVelocityOptimization(tfco.ConstrainedMinimizationProblem):
    """
    This class describes the optimization problem associated with minimizing the Lagrangian metric w.r.t a provided
    waiting state (UAV position radius level - $r_{U}$) and waiting state action (UAV radial velocity - $v_{r}$)
    """

    def __init__(self, r_u, v_r, v_max, p_avg, delta_0, uav_angular_velocity_component):
        """
        The initialization sequence: Parameter Assignments

        :param r_u: The UAV radius level ($r_{U}$)
        :param v_r: The UAV radial velocity component ($v_{r}$)
        :param v_max: The maximum UAV velocity constraint ($V_{\text{max}}$)
        :param p_avg: The average UAV power consumption constraint ($P_{\text{avg}}$)
        :param delta_0: The time-period in which no additional request is received ($\delta_{0}$)
        :param uav_angular_velocity_component: The UAV angular velocity component ($\theta_{c}$)
        """
        super().__init__()
        print('[INFO] ConstrainedOptimization Initialization: Bringing things up...')
        self.v_r = v_r
        self.r_u = r_u
        self.v_max = v_max
        self.p_avg = p_avg
        self.delta_0 = delta_0
        self.theta_c = uav_angular_velocity_component
        # The initialization sequence has been completed

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    @property
    def num_constraints(self):
        """
        Get the number of constraints involved in this optimization [API requirement]

        :return The number of constraints involved in this optimization [1: max velocity constraint]
        """
        return 1

    def objective(self):
        """
        The Lagrangian cost metric evaluation function ($l_{nu}^{*}(s; v_{r})$) [External]

        :return: The objective function (Lagrangian evaluation) that is to be optimized
        """
        r_u, v_r, theta_c__, v_max, p_avg = self.r_u, self.v_r, self.theta_c, self.v_max, self.p_avg
        v, delta_0 = tf.sqrt(tf.add((v_r ** 2), tf.multiply((r_u ** 2), tf.square(theta_c__)))), self.delta_0
        return nu * (evaluate_power_consumption(v) - p_avg) * delta_0

    def constraints(self):
        """
        Define the constraints associated with this optimization problem

        :return: A tensor with the constraints involved in this optimization problem [max velocity constraint]
        """
        r_u, v_r, theta_c__, v_max = self.r_u, self.v_r, self.theta_c, self.v_max
        v = tf.sqrt(tf.add((v_r ** 2), tf.multiply((r_u ** 2), tf.square(theta_c__))))
        return tf.subtract(v, v_max)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the script that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        print('[INFO] ConstrainedOptimization Termination: Tearing things down - Exception Type = {} | '
              'Exception Value = {} | Traceback = {}'.format(exc_type, exc_val, exc_tb))
        # Nothing to do here...


# The secondary variables (non-trainables) in this optimization problem
nu, p_1, p_2, p_3, u_tip, v_0 = 5e-4, 580.65, 790.6715, 0.0073, 200.0, 7.2

# The primary variable (trainable) in this optimization problem
alpha_c, delta_c, th, converged, conf, conf_th = 1e-4, 1e-4, 0, False, 0, 5
theta_c_1, theta_c = tf.Variable(np.inf), tf.Variable(100.0)


def evaluate_power_consumption(v):
    """
    Determine the amount of power consumed by the UAV in Watts ($P_{\text{mob}}(V)$ W)

    :param v: The instantaneous horizontal flying velocity of the UAV in m/s ($v_{u}(t))

    :return: The UAV's power consumption when flying at the specified velocity (or hovering) according to the
             provided motion and power consumption profiles
    """
    return (p_1 * (1 + ((3 * (v ** 2)) / (u_tip ** 2)))) + (p_2 * ((((1 + ((v ** 4) / (4 * (v_0 ** 4)))) ** 0.5) -
                                                                    ((v ** 2) / (2 * (v_0 ** 2))))) ** 0.5) + \
        (p_3 * (v ** 3))


print(f'Hovering Power Consumption: {evaluate_power_consumption(0.0)}')
print(f'Power Consumption when Flying at Power-Minimizing Speed: {evaluate_power_consumption(21.473446)}')

# Run Trigger for Functional Testing
with UAVAngularVelocityOptimization(1000.0, 27.5, 55.0, 1200.0, 4.3456, theta_c) as problem:
    optimizer = tfco.LagrangianOptimizer(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_c),
                                         num_constraints=problem.num_constraints)
    # The list of relevant variables for this optimization problem
    var_list = [theta_c, problem.trainable_variables, optimizer.trainable_variables()]
    while (not converged) or (conf < conf_th):
        tf.compat.v1.assign(theta_c_1, theta_c, validate_shape=True, use_locking=True)
        optimizer.minimize(problem, var_list=var_list)
        if tf.math.is_nan(theta_c):
            th += 1
            tf.compat.v1.assign(theta_c, np.random.random() if (th < 100) else 0.0,
                                validate_shape=True, use_locking=True)
        print(f'{theta_c} | {theta_c_1} | {abs(theta_c - theta_c_1)}')
        converged = (abs(theta_c - theta_c_1) < delta_c)
        conf += 1 if converged else (-1 * conf)  # Update OR Reset if it is a red herring
# The TensorFlow Constrained Optimization utility's functional testing for angular velocity optimization ends here
