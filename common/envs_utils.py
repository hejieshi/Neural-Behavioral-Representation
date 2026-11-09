import numpy as np

def Ry(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def root_frame2(O, A):
    """
    计算基于 O, A 点构建的 Root 坐标系
    
    参数:
        O: 列表或数组 [x, y, z]，原点
        A: 列表或数组 [x, y, z]，确定 Z 轴方向

    返回:
        X_root, Y_root, Z_root: Root坐标系的三个正交基向量
    """
    O = np.array(O, dtype=float)
    A = np.array(A, dtype=float)
    vOA = A - O
    mag_OA = np.linalg.norm(vOA)
    if mag_OA < 1e-06:
        raise ValueError('点 O 和点 A 重合或过近，无法构建坐标系。')
    Z_root = vOA / mag_OA
    Y_raw = np.array([-vOA[1], vOA[0], 0.0])
    mag_Y = np.linalg.norm(Y_raw)
    if mag_Y < 1e-06:
        print('警告: OA 向量完全垂直于世界 XOY 平面。已为其分配默认 Y 轴。')
        Y_root = np.array([0.0, 1.0, 0.0])
    else:
        Y_root = Y_raw / mag_Y
    X_root = np.cross(Y_root, Z_root)
    return np.column_stack((X_root, Y_root, Z_root))

def root_frame2_euler(yaw, pitch):
    O = np.array([0, 0, 0])
    Ax = 1 * np.cos(pitch) * np.cos(yaw)
    Ay = 1 * np.cos(pitch) * np.sin(yaw)
    Az = -1 * np.sin(pitch)
    A = np.array([Ax, Ay, Az])
    return root_frame2(O, A)
