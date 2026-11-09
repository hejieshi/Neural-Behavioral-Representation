import numpy as np
from common.envs_utils import Ry, Rz, root_frame2_euler

def matrix_to_euler_dict_vae(X_hat, keys):
    """
    Convert generated feature vectors back to euler dict.

    Input:
        X_hat : array-like, shape [T, D]
        keys  : edge key order used in training
    Output:
        list of euler dicts, each dict maps edge -> [yaw, pitch, 0]
    """
    X_hat = np.asarray(X_hat)
    num_edges = len(keys)
    out = {}
    for k in keys:
        out[k] = []
    for sample in X_hat:
        ptr = 0
        for k in keys:
            sin_yaw = sample[ptr + 0]
            cos_yaw = sample[ptr + 1]
            sin_pitch = sample[ptr + 2]
            cos_pitch = sample[ptr + 3]
            yaw_norm = np.sqrt(sin_yaw ** 2 + cos_yaw ** 2) + 1e-08
            pitch_norm = np.sqrt(sin_pitch ** 2 + cos_pitch ** 2) + 1e-08
            sin_yaw /= yaw_norm
            cos_yaw /= yaw_norm
            sin_pitch /= pitch_norm
            cos_pitch /= pitch_norm
            yaw = np.arctan2(sin_yaw, cos_yaw)
            pitch = np.arctan2(sin_pitch, cos_pitch)
            pose = [float(yaw), float(pitch), 0.0]
            ptr += 4
            out[k].append(pose)
    return out

def reconstruct_from_feature(eulers, root_eulers, Proot_lower_list, bone_length, body_part):
    idx = {name: i for i, name in enumerate(body_part)}
    T = len(root_eulers)
    T = len(eulers[list(eulers.keys())[0]])
    N = len(body_part)
    recon = np.zeros((T, 3, N))

    def fk_chain(parent, child, R_parent, P_parent, t):
        yaw, pitch, _ = eulers[f'{parent}-{child}'][t]
        R_child = R_parent @ Rz(yaw) @ Ry(pitch)
        length = bone_length[f'{parent}-{child}']
        direction = R_child @ np.array([1, 0, 0])
        P_child = P_parent + direction * length
        return (R_child, P_child)
    for t in range(T):
        joints = {}
        yaw, pitch, _ = root_eulers[t]
        R_root = root_frame2_euler(yaw, pitch)
        P_root = Proot_lower_list[t]
        joints['Sacrum'] = P_root
        R1, P1 = fk_chain('Sacrum', 'Thigh(R)', R_root, joints['Sacrum'], t)
        R2, P2 = fk_chain('Thigh(R)', 'Knee(R)', R1, P1, t)
        R3, P3 = fk_chain('Knee(R)', 'Ankle(R)', R2, P2, t)
        joints['Thigh(R)'] = P1
        joints['Knee(R)'] = P2
        joints['Ankle(R)'] = P3
        R1, P1 = fk_chain('Sacrum', 'Thigh(L)', R_root, joints['Sacrum'], t)
        R2, P2 = fk_chain('Thigh(L)', 'Knee(L)', R1, P1, t)
        R3, P3 = fk_chain('Knee(L)', 'Ankle(L)', R2, P2, t)
        joints['Thigh(L)'] = P1
        joints['Knee(L)'] = P2
        joints['Ankle(L)'] = P3
        R1, P1 = fk_chain('Sacrum', 'Neck', R_root, joints['Sacrum'], t)
        joints['Neck'] = P1
        R_root = R1
        R1, P1 = fk_chain('Neck', 'Shoulder(R)', R_root, joints['Neck'], t)
        R2, P2 = fk_chain('Shoulder(R)', 'Elbow(R)', R1, P1, t)
        R3, P3 = fk_chain('Elbow(R)', 'Wrist(R)', R2, P2, t)
        joints['Shoulder(R)'] = P1
        joints['Elbow(R)'] = P2
        joints['Wrist(R)'] = P3
        R1, P1 = fk_chain('Neck', 'Shoulder(L)', R_root, joints['Neck'], t)
        R2, P2 = fk_chain('Shoulder(L)', 'Elbow(L)', R1, P1, t)
        R3, P3 = fk_chain('Elbow(L)', 'Wrist(L)', R2, P2, t)
        joints['Shoulder(L)'] = P1
        joints['Elbow(L)'] = P2
        joints['Wrist(L)'] = P3
        R1, P1 = fk_chain('Neck', 'Nose', R_root, joints['Neck'], t)
        joints['Nose'] = P1
        for name in joints:
            recon[t, :, idx[name]] = joints[name]
    return recon
