from pathlib import Path
import pybullet_data
import functools
import inspect
import time
import numpy as np
import pybullet

class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None):
        """Creates a Bullet client and connects to a simulation.

    Args:
      connection_mode:
        `None` connects to an existing simulation or, if fails, creates a
          new headless simulation,
        `pybullet.GUI` creates a new simulation with a GUI,
        `pybullet.DIRECT` creates a headless simulation,
        `pybullet.SHARED_MEMORY` connects to an existing simulation.
    """
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT
        self._client = pybullet.connect(connection_mode, options='--background_color_red=0.2 --background_color_green=0.2 --background_color_blue=0.2')

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            if name not in ['invertTransform', 'multiplyTransforms', 'getMatrixFromQuaternion', 'getEulerFromQuaternion', 'computeViewMatrixFromYawPitchRoll', 'computeProjectionMatrixFOV', 'getQuaternionFromEuler']:
                attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute

class Camera:

    def __init__(self, bc, fps=60, dist=2.5, yaw=0, pitch=-5):
        self._p = bc
        self._dist = dist
        self._yaw = yaw
        self._pitch = pitch
        self._coef = np.array([1.0, 1.0, 0.1])
        self._fps = fps
        self._target_period = 1 / fps
        self._counter = time.perf_counter()
        try:
            self.width, self.height = self._p.getDebugVisualizerCamera()[0:2]
        except:
            self.width, self.height = (1024, 768)
        self.lookat((0, 0, 1))

    def track(self, pos, smooth_coef=None):
        try:
            smooth_coef = self._coef if smooth_coef is None else smooth_coef
            assert (smooth_coef <= 1).all(), 'Invalid camera smoothing parameters'
            yaw, pitch, dist, lookat_ = self._p.getDebugVisualizerCamera()[-4:]
            lookat = (1 - smooth_coef) * lookat_ + smooth_coef * pos
            self._p.resetDebugVisualizerCamera(dist, yaw, pitch, lookat)
            self.camera_target = lookat
            self._yaw, self._pitch, self._dist = (yaw, pitch, dist)
            self.wait()
        except:
            pass

    def lookat(self, pos):
        self.camera_target = pos
        self._p.resetDebugVisualizerCamera(self._dist, self._yaw, self._pitch, pos)

    def dump_rgb_array(self):
        lookat = [0, 0, 1]
        distance = 10
        yaw, pitch, _, _ = self._p.getDebugVisualizerCamera()[-4:]
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(lookat, distance, yaw, pitch, 0, upAxisIndex=2)
        _, _, rgb_array, _, _ = self._p.getCameraImage(width=800, height=600, viewMatrix=view_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        import numpy as np
        rgb_array = np.array(rgb_array, dtype=np.uint8).reshape(600, 800, 4)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def wait(self):
        delta = time.perf_counter() - self._counter
        time.sleep(max(self._target_period - delta, 0))
        now = time.perf_counter()
        self._fps = 0.99 * self._fps + 0.01 / (now - self._counter)
        self._counter = now

    def dump_orthographic_rgb_array(self, lookat=[0, 0, 0]):
        distance = 20
        yaw, pitch, _, lookat = self._p.getDebugVisualizerCamera()[-4:]
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(lookat, distance, yaw, pitch, 0, upAxisIndex=2)
        aspect = self.width / self.height
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=20, aspect=aspect, nearVal=0.01, farVal=1000)
        _, _, rgb_array, _, _ = self._p.getCameraImage(width=self.width * 2, height=self.height * 2, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL, flags=pybullet.ER_NO_SEGMENTATION_MASK)
        return rgb_array

class PaperScene:

    def __init__(self, client):
        self._p = client

    def initialize(self):
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor_id = self._p.loadURDF('plane.urdf', [0, 0, 0], useMaximalCoordinates=True, globalScaling=2.0)
        self.ground_plane_mjcf = [floor_id]
        tex_path = Path(__file__).resolve().parent / 'custom_grey_grid.png'
        tex_id = self._p.loadTexture(str(tex_path))
        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
            self._p.changeVisualShape(i, -1, rgbaColor=[1.0, 1.0, 1.0, 0.5], textureUniqueId=tex_id)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 0)
        wall_color = [1.0, 1.0, 1.0, 1.0]
        dist = 60
        wall_height = 30
        t = 0.5
        walls = [([0, dist, wall_height], [dist, t, wall_height]), ([0, -dist, wall_height], [dist, t, wall_height]), ([dist, 0, wall_height], [t, dist, wall_height]), ([-dist, 0, wall_height], [t, dist, wall_height]), ([0, 0, wall_height * 2], [dist, dist, t])]
        for pos, extents in walls:
            wall_shape = self._p.createVisualShape(self._p.GEOM_BOX, halfExtents=extents, rgbaColor=wall_color)
            self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_shape, basePosition=pos)
