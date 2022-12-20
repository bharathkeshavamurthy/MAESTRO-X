"""
This script generates a set of $N$ random trajectories between any two given points $\mathbf{x}_{0}{=}(x_{0}, y_{0})$
and $\mathbf{x}_{M}{=}(x_{M}, y_{M})$.

Author: Bharath Keshavamurthy <bkeshava@purdue.edu>
Organization: School of Electrical & Computer Engineering, Purdue University, West Lafayette, IN.
Copyright (c) 2021. All Rights Reserved.
"""

# API-specific setup for TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The imports
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ThreadPoolExecutor

# tf.debugging.set_log_device_placement(True)

"""
Project Odin (v21.03) Notation
"""


# Core Descriptive Internal Class
class RandomTrajectoriesGeneration(object):
    """
    This class encapsulates the operations associated with generating a set of trajectories (size = swarm_size $N$)
    between two given points (in rectangular coordinates: $\mathbf{x}_{0}$ and $\mathbf{x}_{M}$).
    """

    def __init__(self, source, destination, radial_bounds, angular_bounds, swarm_size, segment_size, interpolation_num):
        """
        The initialization sequence

        v21.03: Circular Cell of radius $a$ meters | 2-dimensional random walks | Optimization via Interpolation

        :param source: The starting point for the random paths that are to be generated
                       $\mathbf{x}_{0}{=}(x_{0}, y_{0})$
        :param destination: The ending point for the random paths that are to be generated
                            $\mathbf{x}_{M}{=}(x_{M}, y_{M})$
        :param radial_bounds: The lower and upper bounds for the allowed radii values $(-a, a)$ [Polar Perspective]
        :param angular_bounds: The lower and upper bounds for the allowed angle values $(0, 2\pi)$ [Polar Perspective]
        :param swarm_size: The swarm size, i.e., the number of random trajectories to be generated [concurrently]
                           between x_0 and x_m ($N$)
        :param segment_size: The resolution of the line-segment (trajectory) generated between x_0 and x_m ($M$)
        :param interpolation_num: The number of points to be inserted/interpolated between any two given points in the
                                  generated random trajectory, during the "optimization via interpolation" phase
        """
        print('[INFO] RandomTrajectoriesGeneration Initialization: Bringing things up...')
        self.x_0, self.x_m = source, destination
        self.r_bounds, self.theta_bounds = radial_bounds, angular_bounds
        self.n, self.m, self.m_ip = swarm_size, segment_size, interpolation_num
        # self.strategy = tf.distribute.MirroredStrategy()
        # The initialization sequence has been completed

    def __enter__(self):
        """
        The enter method for the off-site context manager

        :return: The instance itself
        """
        return self

    def __generate_segment(self, traj):
        """
        Generate a random $M$-segment trajectory

        :param traj: Assign the generated random $M$-segment trajectory to this [indexed] collection [Mx2 tensor]
        """
        (r_min, r_max), (th_min, th_max) = self.r_bounds, self.theta_bounds
        r = tf.random.uniform(shape=[self.m, 1], minval=r_min, maxval=r_max, dtype=tf.float64)
        theta = tf.random.uniform(shape=[self.m, 1], minval=th_min, maxval=th_max, dtype=tf.float64)
        tf.compat.v1.assign(traj, tf.concat([tf.multiply(r, tf.math.cos(theta)),
                                             tf.multiply(r, tf.math.sin(theta))], axis=1),
                            validate_shape=False, use_locking=False)
        # Nothing to return...

    # Core routine
    def generate(self):
        """
        Generate $N$ (self.n) random $M$-segment (self.m) trajectories between $\mathbf{x}_{0}$ (self.x_0) and
        $\mathbf{x}_{M}$ (self.x_m) [Concurrent Ops | Pre-Optimization]

        :return: An (N, M, 2) tensor: $N$ $M$-trajectory segments with each point represented as a 2-tuple (x, y) in
                 rectangular coordinates
        """
        n, m = self.n, self.m
        # The random trajectories' collection initialization
        trajs = tf.Variable(tf.zeros(shape=(n, m, 2), dtype=tf.float64), dtype=tf.float64)
        # Concurrent generation of random trajectories pre-optimization
        # with self.strategy.scope():
        with ThreadPoolExecutor(max_workers=(2 * n)) as executor:
            for i in range(n):
                executor.submit(self.__generate_segment, trajs[i, :])
        return trajs

    def __optimize_segment(self, traj, opt_traj):
        """
        Optimize the randomness for the given $M$-segment trajectory via Scipy UnivariateSpline interpolation

        :param traj: The [indexed] random trajectory generated via uniform sampling in polar coordinates [Mx2 tensor]
        :param opt_traj: The [indexed] new trajectory collection to house the interpolated one [Concurrency Op]
                         [(m_ip * (M + 2))x2 tensor]
        """
        i_s = [_ for _ in range(traj.shape[0] + 2)]  # +2 to account for the concat of source and destination points
        a, x_0, x_m, m_pre, m_ip = self.r_bounds[1], self.x_0, self.x_m, len(i_s), self.m_ip
        x = np.linspace(0, (m_pre - 1), (m_ip * m_pre))
        tf.compat.v1.assign(opt_traj, tf.clip_by_norm(tf.constant(list(zip(
            UnivariateSpline(i_s, tf.concat([x_0[:, 0], traj[:, 0], x_m[:, 0]], axis=0), s=0)(x),
            UnivariateSpline(i_s, tf.concat([x_0[:, 1], traj[:, 1], x_m[:, 1]], axis=0), s=0)(x)))), a, axes=1),
                            validate_shape=False, use_locking=False)
        # Nothing to return...

    def optimize(self, trajs):
        """
        Optimize the randomness via Scipy UnivariateSpline interpolation

        :param trajs: The indexed random trajectories generated via uniform sampling in polar coordinates [NxMx2 tensor]

        :return The new trajectories' collection housing the optimized ones [Nx(m_ip * (M + 2))x2 tensor]
                [Concurrency Op]
        """
        n, m_post = self.n, (self.m_ip * (self.m + 2))
        # The optimized trajectories' collection to house the results of the interpolation phase
        opt_trajs = tf.Variable(tf.zeros(shape=[n, m_post, 2], dtype=tf.float64), dtype=tf.float64)
        # Concurrent optimization via interpolation of the generated random trajectories
        # with self.strategy.scope():
        with ThreadPoolExecutor(max_workers=(2 * n)) as executor:
            for i in range(n):
                executor.submit(self.__optimize_segment, trajs[i, :], opt_trajs[i, :])
        return opt_trajs

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The termination sequence

        :param exc_type: The type of exception raised in the application that caused the code to exit
        :param exc_val: The value or relevant data/information associated with the raised exit-exception
        :param exc_tb: The traceback details of the raised exit-exception
        """
        print(f'[INFO] RandomTrajectoriesGeneration Termination: Tearing things down - Exception Type = {exc_type} | '
              f'Exception Value = {exc_val} | Traceback = {exc_tb}')
        # Nothing to do here...


# Run Trigger
if __name__ == '__main__':
    start_time = time.monotonic()
    # Generator Instance Creation and Optimized [Interpolated] Random Trajectories Generation
    with RandomTrajectoriesGeneration(tf.constant([[10.0, -2.0]], dtype=tf.float64),
                                      tf.constant([[180.0, 900.0]], dtype=tf.float64), (-1000.0, 1000.0),
                                      (0, (2 * np.pi)), 1024, 16, 10) as generator:
        trajectories = generator.optimize(generator.generate())
    stop_time = time.monotonic()
    print(f'[INFO] RandomTrajectoriesGeneration main: Optimized Random Trajectories = {trajectories} | '
          f'Process Time = {stop_time - start_time} seconds')

    # Trajectory Animations
    fig, ax = plt.subplots()
    plt.xlim([-1000.0, 1000.0])
    plt.ylim([-1000.0, 1000.0])
    ax.scatter(10.0, -2.0, color='green', s=20)
    ax.scatter(180.0, 900.0, color='red', s=20)
    for x_m__, y_m__ in trajectories[1020].numpy():
        ax.scatter(x_m__, y_m__, color='blue', s=5)
        plt.pause(5e-15)
    plt.draw()
    plt.show()
    fig.savefig('plots/random_trajectory_generation.png')
    # The evaluation of the RandomTrajectoriesGeneration utility class ends here...
