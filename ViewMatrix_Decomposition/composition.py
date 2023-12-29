import os
import numpy as np
import json

# Finding the config file
config_file_path = r'C:\Users\skvic\computer-vision\ViewMatrix_Decomposition\config.json'

# Reading and decoding the JSON file
with open(config_file_path, 'r') as f:
    config_data = json.load(f)

viewMatrix = np.array(config_data['sensors']['lidar']['sensor_fusion']['cam1']['view_matrix'])

viewMatrix_3by3 = viewMatrix[[0, 1, 3], :3]

m1 = np.matmul(viewMatrix_3by3, viewMatrix_3by3.T)
m2 = m1 / m1[2, 2]

## 1. Intrinsic Decomposition
cx = m2[0, 2]
cy = m2[1, 2]
fx = np.sqrt(m2[0, 0] - cx**2)
fy = np.sqrt(m2[1, 1] - cy**2)
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

viewMatrix_3by4 = viewMatrix[[0, 1, 3], :]
T_map2c = np.vstack([np.linalg.inv(K).dot(viewMatrix_3by4), [0, 0, 0, 1]])

R_wrld_wrt_cam = T_map2c[:3, :3]  # Rotation from world to camera frame
T_wrld_wrt_cam = T_map2c[:3, 3]   # Translation from world to camera frame

# Camera projection matrix (from world to camera frame)
P = K.dot(np.hstack([R_wrld_wrt_cam, T_wrld_wrt_cam[:, np.newaxis]]))

T_wrld2cam = np.vstack([np.hstack([R_wrld_wrt_cam, T_wrld_wrt_cam[:, np.newaxis]]), [0, 0, 0, 1]])

## 2. Extrinsic Decomposition
'''
yaw - pitch - roll
'''
pitch = -np.arcsin(R_wrld_wrt_cam[2, 0])
yaw = np.arctan2(R_wrld_wrt_cam[2, 1], R_wrld_wrt_cam[2, 2])
roll = np.arctan2(R_wrld_wrt_cam[1, 0], R_wrld_wrt_cam[0, 0])

'''
pitch - yaw - roll
'''
pitch = np.arctan2(-R_wrld_wrt_cam[2, 0], np.sqrt(R_wrld_wrt_cam[2, 1]**2 + R_wrld_wrt_cam[2, 2]**2))
yaw = np.arctan2(R_wrld_wrt_cam[1, 0], R_wrld_wrt_cam[0, 0])
roll = np.arctan2(R_wrld_wrt_cam[2, 1], R_wrld_wrt_cam[2, 2])

print(pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi)

# pitch difference 1 degree
pitch += 1 * np.pi/180
R_yaw = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
])

R_pitch = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])

R_roll = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
])

'''
yaw - pitch - roll
'''
R_wrld_wrt_cam_dif = R_yaw @ R_pitch @ R_roll

'''
pitch - yaw - roll
'''
R_wrld_wrt_cam_dif = R_pitch @ R_yaw @ R_roll