#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class Animator:
    def __init__(self, robot, q, pos_ref: np.ndarray = None) -> None:
        self.robot = robot
        self.q = q
        self.frames = q.shape[0]

        self.fig = plt.figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(autoscale_on=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        self.ax.set_aspect('equal')

        if pos_ref is not None:
            self.ax.plot(pos_ref[0], pos_ref[1],'o', color="darkred")

        self.line, = self.ax.plot([], [], 'o-', lw=2, color='k')

    def update(self, i):
        p_joints = self.robot.fk_for_visualization(self.q[i, :])
        x = p_joints[:, 0]
        y = p_joints[:, 2]
        self.line.set_data(x, y)
        return self.line,

    def animate(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, self.frames,
                                            interval=100, blit=True)
        plt.show()
