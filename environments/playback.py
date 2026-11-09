import numpy as np
from pathlib import Path
from environments.viewer_controls import ViewerControls

from common.envs_utils import root_frame2_euler
from environments.kinematics import matrix_to_euler_dict_vae, reconstruct_from_feature


class PoseState:
    def __init__(self, initial_pose, initial_state, body_info):
        pose = np.load(initial_pose, allow_pickle=False)
        if pose.shape != (1, 151):
            raise ValueError('Initial pose must have shape (1, 151).')
        self.pose = pose.astype(np.float32)
        with np.load(initial_state, allow_pickle=False) as state:
            self.facing = float(state['root_facing'].reshape(-1)[0])
            self.pitching = float(state['root_pitching'].reshape(-1)[0])
            self.root = state['root_xyz'].astype(np.float64).reshape(1, 3).copy()
        # body_info contains the original dictionary of bone lengths.
        with np.load(body_info, allow_pickle=True) as body:
            self.bone_length = body['bone_length'].item()
            self.body_part = body['body_part'].copy()
            self.bone_keys = body['bone_keys'].copy()

    def advance(self, pose):
        pose = np.asarray(pose, dtype=np.float32).reshape(1, 151)
        matrix = root_frame2_euler(self.facing, self.pitching)
        self.root += (matrix * pose[:, None, :3]).sum(axis=2)
        self.facing = (self.facing + float(pose[0, 3])) % (2 * np.pi)
        self.pitching = (self.pitching + float(pose[0, 4])) % (2 * np.pi)
        self.pose = pose
        eulers = matrix_to_euler_dict_vae(pose[:, 95:], self.bone_keys)
        root_eulers = np.array([[self.facing, self.pitching, 0.0]])
        joints = reconstruct_from_feature(eulers, root_eulers, self.root,
                                          self.bone_length, self.body_part)
        # Retain the original renderer's coordinate scale.
        return joints.transpose(0, 2, 1) / 20, eulers, root_eulers


class Viewer(ViewerControls):
    def __init__(self, body_part, fps=30, track=True, gui=True, env=None, output_dir=None):
        import pybullet as pb
        from common.bullet_utils import BulletClient, Camera, PaperScene
        from environments.mocap_renderer import MultiMocapCharactersAnimal

        self.client = BulletClient(pb.GUI)
        self._p = self.client
        self.env = env
        self.output_dir = Path(output_dir or 'outputs/play')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_characters = 1
        self.character_index = 0
        self.controller_autonomy = 1.0
        self.debug = False
        self.gui = False
        self.use_params = gui
        self.camera_tracking = track
        self.camera_smooth = np.ones(3)
        self.closed = False
        self.paused = False
        self.video_log = None
        for flag in (pb.COV_ENABLE_GUI, pb.COV_ENABLE_KEYBOARD_SHORTCUTS,
                     pb.COV_ENABLE_MOUSE_PICKING, pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                     pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, pb.COV_ENABLE_RGB_BUFFER_PREVIEW):
            self.client.configureDebugVisualizer(flag, 0)
        self.client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        self.camera = Camera(self.client, fps=fps, dist=6 if track else 12, pitch=-10, yaw=45)
        self.scene = PaperScene(self.client)
        self.scene.initialize()
        edges = [('Sacrum', 'Neck'), ('Neck', 'Nose'),
                 ('Neck', 'Shoulder(L)'), ('Shoulder(L)', 'Elbow(L)'), ('Elbow(L)', 'Wrist(L)'),
                 ('Neck', 'Shoulder(R)'), ('Shoulder(R)', 'Elbow(R)'), ('Elbow(R)', 'Wrist(R)'),
                 ('Sacrum', 'Thigh(L)'), ('Thigh(L)', 'Knee(L)'), ('Knee(L)', 'Ankle(L)'),
                 ('Sacrum', 'Thigh(L)'), ('Thigh(L)', 'Knee(L)'), ('Knee(L)', 'Ankle(L)'),
                 ('Sacrum', 'Thigh(R)'), ('Thigh(R)', 'Knee(R)'), ('Knee(R)', 'Ankle(R)')]
        self.characters = MultiMocapCharactersAnimal(
            self.client, 1, np.array([[0.376, 0.490, 0.545, 1.0]]),
            skeleton_edges=edges, body_part=body_part)
        self.client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        if self.use_params:
            self._setup_debug_parameters()

    def render(self, joints, root_xyzs=None, action=None):
        if self.closed or not self.client.isConnected():
            return False
        self.root_xyzs = joints[:, 8, :] if root_xyzs is None else root_xyzs
        self.characters.set_joint_positions(joints[0], 0)
        if self.debug:
            print(f'FPS: {self.camera._fps:4.1f} | Time Left: 0.0 | Distance: 0.0')
            if action is not None:
                print(f'max: {float(action.max()):.2f} | mean: {float(action.mean()):.2f} | '
                      f'median: {float(action.median()):.2f} | min: {float(action.min()):.2f}')
        self._handle_mouse_press()
        self._handle_key_press()
        while self.paused and not self.closed and self.client.isConnected():
            self._handle_key_press()
            if not self.closed:
                self.camera.wait()
        if self.closed or not self.client.isConnected():
            return False
        if self.use_params:
            self._handle_parameter_update()
        if self.camera_tracking:
            self.camera.track(np.array([3, 1, 3]), self.camera_smooth)
        else:
            self.camera.wait()
        return True

    def reset(self):
        self.env.reset()

    def _handle_key_press(self, keys=None):
        from datetime import datetime
        from imageio.v2 import imwrite
        keys = self.client.getKeyboardEvents() if keys is None else keys
        released = self.client.KEY_WAS_RELEASED
        for key, event in keys.items():
            if not event & released:
                continue
            if key == ord('d'):
                self.debug = not self.debug
            elif key == ord('g'):
                self.gui = not self.gui
                self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, int(self.gui))
            elif key == ord('n'):
                self.character_index = (self.character_index + 1) % self.num_characters
                self.camera.lookat(self.root_xyzs[self.character_index])
            elif key == ord('m'):
                self.camera_tracking = not self.camera_tracking
            elif key == ord('q'):
                self.close()
                return
            elif key == ord('r'):
                self.reset()
                self.paused = False
            elif key == ord('t') and hasattr(self.env, 'reset_target'):
                self.env.reset_target()
            elif key in (ord('i'), ord('a')):
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                image = (self.camera.dump_rgb_array() if key == ord('i')
                         else self.camera.dump_orthographic_rgb_array())
                imwrite(self.output_dir / f'{stamp}_{chr(key)}.png', image)
            elif key == ord('v'):
                if self.video_log is None:
                    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    self.video_log = self.client.startStateLogging(
                        self.client.STATE_LOGGING_VIDEO_MP4, str(self.output_dir / f'{stamp}.mp4'))
                else:
                    self.client.stopStateLogging(self.video_log)
                    self.video_log = None
            elif key == ord(' '):
                self.paused = not self.paused

    def close(self):
        if self.client.isConnected():
            if self.video_log is not None:
                self.client.stopStateLogging(self.video_log)
                self.video_log = None
            self.client.disconnect()
        self.closed = True
