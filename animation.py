#!/usr/bin/env python3

from re import M
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import pinocchio as pin
from pinocchio.visualize import Panda3dVisualizer
import time


# from sklearn.semi_supervised import SelfTrainingClassifier


class Animator:
    def __init__(self, robot, q, pos_ref: np.ndarray = None) -> None:
        self.robot = robot
        self.q = q
        self.frames = q.shape[0]

        self.fig = plt.figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(autoscale_on=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        self.ax.set_aspect('equal')

        if pos_ref is not None:
            self.ax.plot(pos_ref[0], pos_ref[1], 'o', color="darkred")

        self.line, = self.ax.plot([], [], 'o-', lw=2, color='k')

    def update(self, i):
        p_joints = self.robot.fk_for_visualization(self.q[i, :])
        x = p_joints[:, 0]
        y = p_joints[:, 2]
        self.line.set_data(x, y)
        return self.line,

    def play(self):
        self.anim = animation.FuncAnimation(self.fig, self.update, self.frames,
                                            interval=100, blit=True)
        plt.show()

    def animate(self):
        f = r"animation.mp4"
        writervideo = animation.FFMpegWriter(fps=20)
        self.anim = animation.FuncAnimation(self.fig, self.update, self.frames,
                                            interval=100, blit=True, repeat=True)
        self.anim.save(f, writer=writervideo)
        plt.show()


class Panda3dAnimator:
    def __init__(self, urdf_path, ts, q) -> None:
        """
        :parameter ts: sampling time
        :parameter q: [ns x nj] joint states, nj is number of joints
                and ns is number of samples (length of trajectory)
        """
        self.ts = ts
        self.q = q

        # Let pinocchio to build model, collision and visual models from URDF
        m, cm, vm = pin.buildModelsFromUrdf(urdf_path)

        # Instantiate panda3visualizer
        self.viz = Panda3dVisualizer(m, cm, vm)

    def play(self, k: int = 5):
        """
        :parameter k: number of times to play the trajectory
        """
        self.viz.initViewer()
        self.viz.loadViewerModel(group_name='flexible_arm')

        for _ in range(5):
            self.viz.display(self.q[0, :])
            time.sleep(2)
            for qk in self.q[1:, :]:
                self.viz.display(qk)
                time.sleep(self.ts)
            time.sleep(1)
