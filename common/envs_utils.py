import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np



def Rx(t):
    # roll
    c=np.cos(t); s=np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Ry(t):
    # pitch
    c=np.cos(t); s=np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def Rz(t):
    # yaw
    c=np.cos(t); s=np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def make_T(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T


def root_frame(O,A,P):

    x = A-O
    x /= np.linalg.norm(x)

    temp = P-O
    temp /= np.linalg.norm(temp)

    z = np.cross(x,temp)
    z /= np.linalg.norm(z)

    y = np.cross(z,x)

    return np.column_stack((x,y,z))

def root_frame2(O,A):

    O = np.array(O, dtype=float)
    A = np.array(A, dtype=float)
    
    vOA = A - O
    mag_OA = np.linalg.norm(vOA)
    if mag_OA < 1e-6:
        raise ValueError("ERROR")
    Z_root = vOA / mag_OA


    Y_raw = np.array([-vOA[1], vOA[0], 0.0])
    mag_Y = np.linalg.norm(Y_raw)


    if mag_Y < 1e-6:
        Y_root = np.array([0.0, 1.0, 0.0]) #  Y 
    else:
        Y_root = Y_raw / mag_Y

    X_root = np.cross(Y_root, Z_root)
    return np.column_stack((X_root,Y_root,Z_root))


def root_frame2_euler(yaw,pitch):

    O = np.array([0,0,0])
    Ax = 1 * np.cos(pitch)*np.cos(yaw)
    Ay = 1 * np.cos(pitch)*np.sin(yaw)
    Az =-1 * np.sin(pitch)
    A = np.array([Ax,Ay,Az])

    return root_frame2(O, A)



def vec_to_euler(v):
    if np.linalg.norm(v) < 1e-8:
        return 0,0

    v = v/np.linalg.norm(v)

    v = np.where(np.abs(v) < 1e-8, 0, v)   
    
    yaw = np.arctan2(v[1], v[0])   # z 
    yaw = 0 if abs(yaw) < 1e-8 else yaw
    # pitch = -np.arcsin(v[2])  # y
    pitch = np.arctan2(-v[2], np.sqrt(v[0]*v[0] + v[1]*v[1]))
    pitch = 0 if abs(pitch) < 1e-8 else pitch


    return yaw,pitch


def solve_joint(v,R_parent):

    yaw,pitch = vec_to_euler(R_parent.T @ v)
    R_child = R_parent @ Rz(yaw) @ Ry(pitch)

    return yaw,pitch,R_child

