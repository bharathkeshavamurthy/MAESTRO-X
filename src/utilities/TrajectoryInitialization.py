"""
This utility script tests the initialization process of trajectories in our optimization framework.

Initialization Heuristic: A mixture of lines and random paths.

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
import traceback
import numpy as np
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

"""
Configurations-II: Simulation parameters
"""
pi = np.pi
np.random.seed(6)
a, m, m_ip, n = 1e3, 30, 2, 128
r_bounds, th_bounds = (-a, a), (0, 2 * pi)
m_post, x_g, num_workers = (m + 2) * m_ip, tf.constant([[-570.0, 601.0]], dtype=tf.float64), 1024
x_0, x_m = tf.constant([[400.0, -300.0]], dtype=tf.float64), tf.constant([[-387.50, 391.50]], dtype=tf.float64)

"""
Core operations
"""


# noinspection PyMethodMayBeStatic
class RandomTrajectoriesGeneration(object):

    def __enter__(self):
        return self

    def __generate(self, traj):
        (r_min, r_max), (th_min, th_max) = r_bounds, th_bounds
        r = tf.random.uniform(shape=[m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)

        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)),
                                             tf.multiply(r, tf.math.sin(theta))],
                                            axis=1), validate_shape=True, use_locking=True)

    def generate(self, n_w):
        trajs = tf.Variable(tf.zeros(shape=(int(n / 2), m, 2), dtype=tf.float64), dtype=tf.float64)

        with ThreadPoolExecutor(max_workers=n_w) as executor:
            [executor.submit(self.__generate, trajs[i, :]) for i in range(int(n / 2))]

        return trajs

    def __optimize(self, traj, opt_traj):
        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), (m_ip * len(i_s)), dtype=np.float64)

        tf.compat.v1.assign(opt_traj, tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
                            validate_shape=True, use_locking=True)

    def optimize(self, trajs, n_w):
        opt_trajs = tf.Variable(tf.zeros(shape=[int(n / 2), m_post, 2], dtype=tf.float64), dtype=tf.float64)
        with ThreadPoolExecutor(max_workers=n_w) as executor:
            [executor.submit(self.__optimize, trajs[i, :], opt_trajs[i, :]) for i in range(int(n / 2))]
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] RandomTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


# noinspection PyMethodMayBeStatic
class DeterministicTrajectoriesGeneration(object):

    def __enter__(self):
        return self

    def generate(self):
        x_mid = tf.divide(tf.add(x_0, x_m), 2)
        x_mid_0 = tf.divide(tf.add(x_0, x_mid), 2)
        x_mid_m = tf.divide(tf.add(x_mid, x_m), 2)
        traj = tf.concat([x_mid_0, x_mid, x_mid_m], axis=0)

        i_s = [_ for _ in range(traj.shape[0] + 2)]
        x = np.linspace(0, (len(i_s) - 1), m_post, dtype=np.float64)

        return tf.tile(tf.expand_dims(tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
            axis=0), multiples=[int(n / 2), 1, 1])

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            print(f'[ERROR] DeterministicTrajectoriesGeneration Termination: Tearing things down - '
                  f'Error Type = {exc_type} | Error Value = {exc_val} | Traceback = {traceback.print_tb(exc_tb)}.')


# Run Trigger
if __name__ == '__main__':
    r_gen, d_gen = RandomTrajectoriesGeneration(), DeterministicTrajectoriesGeneration()

    d_trajs = d_gen.generate()
    r_trajs = r_gen.optimize(r_gen.generate(num_workers), num_workers)

    combined_trajs = tf.concat([d_trajs, r_trajs], axis=0)

    tf.io.write_file(str(uuid.uuid4()),
                     tf.strings.format('{}\n', tf.constant(str(combined_trajs.numpy()), dtype=tf.string)))
