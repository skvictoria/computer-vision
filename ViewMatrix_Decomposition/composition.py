import os
import numpy as np
import json

# Finding the config file
config_file_path = os.path.join('config.json')

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