from pyquaternion import Quaternion
import numpy as np

def quaternion_to_rotation_matrix_using_lib(quaternion):
    return Quaternion(quaternion).rotation_matrix

def quaternion_to_rotation_matrix(q):
    # Normalize the quaternion
    q = q / np.linalg.norm(q)
    # Extract the values from Q
    w, x, y, z = q
    # Compute the rotation matrix
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

quaternion = np.array([-0.70276998, -0.00060261, 0.01316346, 0.71129510])

# 1. Given quaternion to rotation matrix
rotation_matrix = quaternion_to_rotation_matrix(quaternion)
print(rotation_matrix)

## 2. Inverse quaternion
camera_to_lidar_quaternion = quaternion
lidar_to_camera_quat = camera_to_lidar_quaternion * np.array([1, -1, -1, -1])  # Invert x, y, z for the inverse
lidar_to_camera_rotation_matrix = quaternion_to_rotation_matrix(lidar_to_camera_quat)

# Invert the translation vector for lidar-to-camera translation
lidar_to_camera_translation = -lidar_to_camera_rotation_matrix.dot(camera_to_lidar_translation)

# Now, define the axis alignment matrix from the lidar frame to the camera frame
axis_alignment = np.array([
    [1, 0, 0],  # Lidar's x (Right) is Camera's x (Right)
    [0, 0, -1], # Lidar's z (Up) is Camera's -y (Down)
    [0, 1, 0]   # Lidar's y (Front) is Camera's z (Front)
])

# Combine the axis alignment with the lidar-to-camera rotation matrix
aligned_rotation_matrix = axis_alignment.dot(lidar_to_camera_rotation_matrix)

# Construct the full transformation matrix, combining rotation and translation
lidar_to_camera_transformation_matrix = np.vstack((
    np.hstack((aligned_rotation_matrix, lidar_to_camera_translation.reshape(-1, 1))),
    [0, 0, 0, 1]
))

print(lidar_to_camera_transformation_matrix)