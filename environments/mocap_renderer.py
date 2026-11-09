import numpy as np
import pybullet as pb
from common.bullet_objects import VSphere, VCapsule
FADED_ALPHA = 1.0

class MultiMocapCharactersAnimal:

    def __init__(self, bc, num_characters, colours=None, links=True, skeleton_edges=None, body_part=None):
        self._p = bc
        self.num_joints = 15
        self.ids = []
        for _ in range(num_characters):
            joint_first = VSphere(bc, radius=0.3, max=True, replica=1)
            joints_rest = VSphere(bc, radius=0.07, max=True, replica=self.num_joints - 1)
            self.ids.extend(joint_first.ids)
            self.ids.extend(joints_rest.ids)
        self.has_links = links
        if links:
            self.linked_joints = []
            idx = {name: i for i, name in enumerate(body_part)}
            for edge in skeleton_edges:
                parent, child = edge
                i = idx[parent]
                j = idx[child]
                self.linked_joints.append([i, j])
            self.linked_joints = np.array(self.linked_joints)
            self.links = {i: [VCapsule(self._p, radius=0.06, height=0.1, rgba=colours[i]) for _ in range(self.linked_joints.shape[0])] for i in range(num_characters)}
            self.z_axes = np.zeros((self.linked_joints.shape[0], 3))
            self.z_axes[:, 0] = 1
            self.heads = [VSphere(bc, radius=0.0002) for _ in range(num_characters)]
        if colours is not None:
            self.colours = colours
            monkey_base_color = [0.55, 0.35, 0.15, 1.0]
            monkey_face_color = [0.75, 0.55, 0.4, 1.0]
            # monkey_base_color = [0.20, 0.40, 0.75, 1.0]  # blue
            # monkey_face_color = [0.50, 0.75, 0.95, 1.0]  # blue
            for index in range(num_characters):
                self.set_colour(monkey_base_color, index)
                first_joint_id = self.ids[index * self.num_joints]
                self._p.changeVisualShape(first_joint_id, -1, rgbaColor=monkey_face_color)
                if links:
                    self.heads[index].set_color(monkey_face_color)
        bar_A = np.array([0, 10, 15]) / 20
        bar_B = np.array([70, 10, 15]) / 20
        self.add_cylinder_between_points(p_A=bar_A, p_B=bar_B)
        bar_A = np.array([0, 12, 15]) / 20
        bar_B = np.array([70, 12, 15]) / 20
        self.add_cylinder_between_points(p_A=bar_A, p_B=bar_B)
        bar_A = np.array([0, 10, 70]) / 20
        bar_B = np.array([70, 10, 70]) / 20
        self.add_cylinder_between_points(p_A=bar_A, p_B=bar_B)
        bar_A = np.array([0, 50, 90]) / 20
        bar_B = np.array([70, 50, 90]) / 20
        self.add_cylinder_between_points(p_A=bar_A, p_B=bar_B)
        bar_A = np.array([0, 90, 70]) / 20
        bar_B = np.array([70, 90, 70]) / 20
        self.add_cylinder_between_points(p_A=bar_A, p_B=bar_B)

    def set_joint_positions(self, xyzs, index):
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        xyzs = np.array(xyzs, copy=True)
        start = index * self.num_joints
        joint_ids = self.ids[start:start + self.num_joints]
        mid_point = 0.5 * (xyzs[0, :] + xyzs[1, :])
        xyzs[0, :] = mid_point
        v_head = xyzs[0, :] - xyzs[1, :]
        for xyz, id in zip(xyzs, joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))
        if self.has_links:
            rgba = self.colours[index].copy()
            rgba[-1] = FADED_ALPHA
            alpha = FADED_ALPHA
            deltas = xyzs[self.linked_joints[:, 1]] - xyzs[self.linked_joints[:, 0]]
            heights = np.linalg.norm(deltas, axis=-1)
            positions = xyzs[self.linked_joints].mean(axis=1)
            OA_batch = deltas
            N = OA_batch.shape[0]
            norms = np.linalg.norm(OA_batch, axis=1, keepdims=True)
            norms[norms < 1e-09] = 1.0
            u = OA_batch / norms
            ux = u[:, 0]
            uy = u[:, 1]
            uz = u[:, 2]
            quats = np.zeros((N, 4))
            s = np.sqrt(np.maximum(0, 2.0 * (1.0 + uz)))
            safe_mask = s > 1e-06
            quats[safe_mask, 0] = -uy[safe_mask] / s[safe_mask]
            quats[safe_mask, 1] = ux[safe_mask] / s[safe_mask]
            quats[safe_mask, 2] = 0.0
            quats[safe_mask, 3] = s[safe_mask] / 2.0
            extreme_neg_mask = uz < -0.999999
            quats[extreme_neg_mask] = [1.0, 0.0, 0.0, 0.0]
            extreme_pos_mask = uz > 0.999999
            quats[extreme_pos_mask] = [0.0, 0.0, 0.0, 1.0]
            for lid, (delta, height, pos, orn, link) in enumerate(zip(deltas, heights, positions, quats, self.links[index])):
                self._p.removeBody(link.id[0])
                if lid == 1:
                    part_rgba = [0.85, 0.65, 0.5, alpha]
                    # part_rgba = [0.20, 0.40, 0.75, 1.0] # blue
                elif lid in [4, 7, 10, 13, 16]:
                    part_rgba = [0.25, 0.2, 0.15, alpha]
                    # part_rgba = [0.10, 0.25, 0.50, 1.0] # blue
                else:
                    part_rgba = [0.6, 0.4, 0.2, alpha]
                    # part_rgba = [0.20, 0.40, 0.75, 1.0] # blue
                if lid == 0:
                    link = VCapsule(self._p, radius=0.24, height=height, rgba=part_rgba)
                elif lid == 1:
                    link = VCapsule(self._p, radius=0.16, height=height, rgba=part_rgba)
                else:
                    link = VCapsule(self._p, radius=0.12, height=height, rgba=part_rgba)
                self.links[index][lid] = link
                link.set_position(pos, orn)
            self.heads[index].set_position(0.5 * (xyzs[4] - xyzs[3]) + xyzs[4])
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        self._p.configureDebugVisualizer(lightPosition=[0, 1, 5])

    def set_colour(self, colour, index):
        start = index * self.num_joints
        joint_ids = self.ids[start:start + self.num_joints]
        for id in joint_ids:
            self._p.changeVisualShape(id, -1, rgbaColor=colour)

    def add_cylinder_between_points(self, p_A, p_B, radius=0.1, rgba=[0.9, 0.9, 0.9, 1.0]):
        import numpy as np
        A = np.array(p_A)
        B = np.array(p_B)
        delta = B - A
        length = np.linalg.norm(delta)
        if length < 1e-05:
            return -1
        mid_point = (A + B) / 2.0
        direction = delta / length
        z_axis = np.array([0.0, 0.0, 1.0])
        dot = np.dot(z_axis, direction)
        if dot > 0.999999:
            orn = [0, 0, 0, 1]
        elif dot < -0.999999:
            orn = [1, 0, 0, 0]
        else:
            cross = np.cross(z_axis, direction)
            s = np.sqrt((1.0 + dot) * 2.0)
            orn = [cross[0] / s, cross[1] / s, cross[2] / s, s / 2.0]
        visual_id = self._p.createVisualShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
        body_id = self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=mid_point, baseOrientation=orn)
        return body_id
