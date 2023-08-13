#!/usr/bin/env python3
from panda3d_viewer import Viewer, ViewerConfig
import numpy as np
import pinocchio as pin
from pinocchio.visualize import Panda3dVisualizer
import time


PANDA3D_CONFIG = ViewerConfig()
PANDA3D_CONFIG.enable_antialiasing(True, multisamples=4)
PANDA3D_CONFIG.enable_shadow(False)
PANDA3D_CONFIG.show_axes(True)
PANDA3D_CONFIG.show_grid(True)
PANDA3D_CONFIG.show_floor(True)
PANDA3D_CONFIG.enable_spotlight(False)
PANDA3D_CONFIG.enable_hdr(True)


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
        self.viewer = Viewer(config=PANDA3D_CONFIG)
        self.viewer.set_background_color((255,255,255))
        # self.viewer.reset_camera()
        # self.viz = RVizVisualizer(m, cm, vm)

    def play(self, k: int = 5):
        """
        :parameter k: number of times to play the trajectory
        """
        self.viz.initViewer(viewer=self.viewer)
        self.viz.loadViewerModel(group_name='flexible_arm')

        for _ in range(k):
            self.viz.display(self.q[0, :])
            time.sleep(2)
            for qk in self.q[1:, :]:
                self.viz.display(qk)
                time.sleep(self.ts)
            time.sleep(1)


class FlexibleArmVisualizer:
    def __init__(self, urdf_path: str, dt: float) -> None:
        """ 
        :param urdf_path: path to a urdf file of a robot
        :param dt: simulation step size
        """
        self.dt = dt

        # Let pinocchio to build model, collision and visual models from URDF
        self.m, self.cm, self.vm = pin.buildModelsFromUrdf(urdf_path)

    def visualize_trajectory(
        self, 
        q_traj: np.ndarray, 
        p_goal: np.ndarray = None,
        n_replays: int = 3
    ):
        n_repeats = q_traj.shape[0]
        q_goal = self._process_goal_position(p_goal)
        q_goal_traj = np.repeat(q_goal.T, n_repeats, axis=0)

        # Concat robot configuratiuon and goal configuration
        q_all = np.hstack((q_traj, q_goal_traj))

        # Play trajectory
        viz = self._start_visualizer()
        self._play(viz, q_all, n_replays)

    def visualize_configuration(
        self,
        q: np.ndarray,
        p_goal: np.ndarray = None
    ):
        t_viz = 5
        n_repeats = int(t_viz/self.dt)
        n_replays = 1

        q_goal = self._process_goal_position(p_goal)
        q_goal_traj = np.repeat(q_goal.T, n_repeats, axis=0)
        q_traj = np.repeat(q.T, n_repeats, axis=0)
        q_all = np.hstack((q_traj, q_goal_traj))

        # Play static trajectory
        viz = self._start_visualizer()
        self._play(viz, q_all, n_replays)        

    @staticmethod
    def _process_goal_position(p_goal):
        """
        :return: full goal joint pose -- position and qquaternion
        """
        # If the goal position is not given set it to z = -2m, so that
        # it gets out of enivorment
        if p_goal is None:
            q_goal = np.array([[0., 0., -2., 0., 0., 0., 0.]]).T
        else:
            # Since the goal is connected to the world through floating joint
            # need to specify position and the quaterion
            quat_goal = np.zeros((4,1))
            q_goal = np.vstack((p_goal, quat_goal))
        return q_goal
    
    def _start_visualizer(self):
        viewer = Viewer(config=PANDA3D_CONFIG)
        viewer.set_background_color(((255, 255, 255)))
        # viewer.reset_camera((0., -2, 0.75), look_at=(0.,1.25,0))

        # Instantiate panda3visualizer
        viz = Panda3dVisualizer(self.m, self.cm, self.vm)
        viz.initViewer(viewer=viewer)
        viz.loadViewerModel(group_name=f'flexible_arm')
        return viz

    def _play(self, viz, q, n_replays):
        for _ in range(n_replays):
            viz.display(q[0, :])
            time.sleep(1)
            viz.play(q[1:, :], self.dt)
            time.sleep(1)
        viz.viewer.stop()
    
    
class Panda3dRenderer:
    def __init__(self, urdf_path: str) -> None:
        # Let pinocchio to build model, collision and visual models from URDF
        m, cm, vm = pin.buildModelsFromUrdf(urdf_path)

        # Instantiate panda3visualizer
        viewer = Viewer(config=PANDA3D_CONFIG)
        viewer.set_background_color(((255, 255, 255)))
        viewer.reset_camera((4., 1.5, 1.5), look_at=(0.,0.,0))
        self.viz = Panda3dVisualizer(m, cm, vm)
        self.viz.initViewer(viewer=viewer)
        self.viz.loadViewerModel(group_name="flexible_arm")

    def draw_sphere(self, pos: np.ndarray) -> None:
        """
        :parameter pos: [3,] position of the sphere
        """
        self.viz.viewer.append_group("sphere_group", remove_if_exists=True)
        self.viz.viewer.append_sphere(
            root_path="sphere_group", name="sphere", radius=0.025
        )
        self.viz.viewer.set_material(
            root_path="sphere_group", name="sphere", color_rgba=(0.87, 0.275, 0.09, 1.0)
        )
        self.viz.viewer.move_nodes(
            root_path="sphere_group",
            name_pose_dict={"sphere": (tuple(pos), (1, 0, 0, 0))},
        )

    def render(self, q: np.ndarray) -> np.ndarray:
        """
        :parameter q: [nj,] joint states, nj is number of joints
        """
        self.viz.display(q)
        img = self.viz.viewer.get_screenshot()
        return img
