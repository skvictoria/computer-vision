import numpy as np
import json
import cv2

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

pitch = np.arctan2(-R_wrld_wrt_cam[2, 0], np.sqrt(R_wrld_wrt_cam[2, 1]**2 + R_wrld_wrt_cam[2, 2]**2))
yaw = np.arctan2(R_wrld_wrt_cam[1, 0], R_wrld_wrt_cam[0, 0])
roll = np.arctan2(R_wrld_wrt_cam[2, 1], R_wrld_wrt_cam[2, 2])

# pitch difference 1 degree
pitch_degree = pitch * 180 / np.pi
pitch_degree += 1
pitch = pitch_degree * np.pi / 180

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

R_wrld_wrt_cam_dif = R_pitch @ R_yaw @ R_roll

print(R_wrld_wrt_cam_dif)
print(R_wrld_wrt_cam)

P_diff = K.dot(np.hstack([R_wrld_wrt_cam_dif, T_wrld_wrt_cam[:, np.newaxis]]))

P = K.dot(np.hstack([R_wrld_wrt_cam, T_wrld_wrt_cam[:, np.newaxis]]))

image = np.zeros((2400, 2080, 3), dtype=np.uint8)

## 1. extrinsic 1 degree -> pixel difference
for y in range(10, 50):
    for x in range(5, 6):
        wu,wv,w = P @ (np.array([x,y,0,1]).T)
        #print(wu/w, wv/w)
        cv2.circle(image, (int(wu/w), int(wv/w)), 5, (0,0,255), -1)

for y in range(10, 50):
    for x in range(5, 6):
        wu,wv,w = P_diff @ (np.array([x,y,0,1]).T)
        #print(wu/w, wv/w)
        cv2.circle(image, (int(wu/w), int(wv/w)), 5, (255,0,0), -1)

cv2.imwrite("png.png", image)

## 2. how many pixel difference?
x = 5
for y in range(10, 50):
    wu_dif, wv_dif, w_dif = P_diff @ (np.array([x,y,0,1]).T)
    wu,wv,w = P @ (np.array([x,y,0,1]).T)
    u_dif = int(wu_dif/w_dif)
    v_dif = int(wv_dif/w_dif)
    u = int(wu/w)
    v = int(wv/w)

    #print(abs(u_dif - u), abs(v_dif - v))

## 3. 1 pixel diff -> ? m diff?
for u in range(100, 101):
    for v in range(2000, 2200):
        wx, wy, _, w = np.linalg.inv(viewMatrix) @ np.array([u, v, 1, 1])
        wx_diff, wy_diff, _, w_diff = np.linalg.inv(viewMatrix) @ np.array([u+8, v, 1, 1])

        print(abs(wx/w) - abs(wx_diff/w_dif), abs(wy/w) - abs(wy_diff/w_dif))