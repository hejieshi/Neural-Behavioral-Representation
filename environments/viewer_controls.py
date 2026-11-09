import numpy as np
DEG2RAD = np.pi / 180
FOOT2METER = 0.3048

class ViewerControls:

    def _setup_debug_parameters(self):
        max_frame = self.env.mocap_data.shape[0] - self.env.num_condition_frames
        self.parameters = [{'default': -1, 'args': ('Start Frame', -1, max_frame, -1), 'dest': (self.env, 'debug_frame_index'), 'func': lambda x: int(x), 'post': lambda: self.env.reset()}, {'default': self.env.data_fps, 'args': ('Target FPS', 1, 240, self.env.data_fps), 'dest': (self.camera, '_target_period'), 'func': lambda x: 1 / (x + 1)}, {'default': 1, 'args': ('Controller Autonomy', 0, 1, 1), 'dest': (self, 'controller_autonomy'), 'func': lambda x: x}, {'default': 1, 'args': ('Camera Track Character', 0, 1, int(self.camera_tracking)), 'dest': (self, 'camera_tracking'), 'func': lambda x: x > 0.5}]
        if self.num_characters > 1:
            self.parameters.append({'default': 1, 'args': ('Selected Character', 1, self.num_characters + 0.999, 1), 'dest': (self, 'character_index'), 'func': lambda x: int(x - 1.001)})
        max_frame_skip = self.env.pose_vae_model.num_future_predictions
        if max_frame_skip > 1:
            self.parameters.append({'default': 1, 'args': ('Frame Skip', 1, max_frame_skip + 0.999, self.env.frame_skip), 'dest': (self.env, 'frame_skip'), 'func': lambda x: int(x)})
        if hasattr(self.env, 'target_direction'):
            self.parameters.append({'default': 0, 'args': ('Target Direction', 0, 359, 0), 'dest': (self.env, 'target_direction'), 'func': lambda x: x / 180 * np.pi, 'post': lambda: self.env.reset_target()})
        if hasattr(self.env, 'target_speed'):
            self.parameters.append({'default': 0, 'args': ('Target Speed', 0.0, 0.8, 0.5), 'dest': (self.env, 'target_speed'), 'func': lambda x: x})
        for param in self.parameters:
            param['id'] = self._p.addUserDebugParameter(*param['args'])

    def _handle_parameter_update(self):
        for param in self.parameters:
            func = param['func']
            value = func(self._p.readUserDebugParameter(param['id']))
            cur_value = getattr(*param['dest'], param['default'])
            if cur_value != value:
                setattr(*param['dest'], value)
                if 'post' in param:
                    post_func = param['post']
                    post_func()

    def _handle_mouse_press(self):
        events = self._p.getMouseEvents()
        for ev in events:
            if ev[0] == 2 and ev[3] == 0 and (ev[4] == self._p.KEY_WAS_RELEASED):
                width, height, _, proj, _, _, _, _, yaw, pitch, dist, target = self._p.getDebugVisualizerCamera()
                pitch *= DEG2RAD
                yaw *= DEG2RAD
                R = np.reshape(self._p.getMatrixFromQuaternion(self._p.getQuaternionFromEuler([pitch, 0, yaw])), (3, 3))
                camera_up = np.matmul(R, [0, 0, 1])
                camera_forward = np.matmul(R, [0, 1, 0])
                camera_right = np.matmul(R, [1, 0, 0])
                x = ev[1] / width
                y = ev[2] / height
                d = proj[5]
                A = target - camera_forward * dist
                aspect = height / width
                B = A + camera_forward * d + (x - 0.5) * 2 * camera_right / aspect - (y - 0.5) * 2 * camera_up
                C = np.array([(B[2] * A[0] - A[2] * B[0]) / (B[2] - A[2]), (B[2] * A[1] - A[2] * B[1]) / (B[2] - A[2]), 0]) / FOOT2METER
                if hasattr(self.env, 'reset_target'):
                    self.env.reset_target(location=C)
