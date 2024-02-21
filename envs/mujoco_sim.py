import mujoco
import mujoco.viewer
import cv2
import numpy as np


def create_video(frames: list, framerate: int, video_name: str):
    height, width, layers = frames[0].shape

    output_video = f'{video_name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(output_video, fourcc, framerate, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


class PDController:
    def __init__(self, Kp, Kd) -> None:
        self.Kp = Kp
        self.Kd = Kd

    def __call__(self, q, dq, q_ref):
        return self.Kp @ (q_ref - q) - self.Kd @ dq


def main(name='flexible_arm'):
    # Load model
    xml_path = 'models/flexible_robot_arm.xml'
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=480, width=640)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)


    # Simulation parameters
    duration = 30 # seconds
    framerate = 60 # Hz
    frames = []

    # Controller
    n_seg = 10
    qa_idxs = [0, 1, 2 + n_seg]
    Kp = np.diag([15, 15, 10])
    Kd = 0.12*np.ones(3)
    q_ref = np.array([0, np.pi/3, np.pi/4])
    pd_ctrl = PDController(Kp, Kd)

    while data.time < duration:
        q = np.array(data.qpos[qa_idxs])
        dq = np.array(data.qvel[qa_idxs])
        data.ctrl = pd_ctrl(q, dq, q_ref)
        mujoco.mj_step(model, data)

        # Update renderer
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)


    # Create video
    video_name = f'{name}_video'
    create_video(frames, framerate, video_name)

if __name__ == '__main__':
    main()