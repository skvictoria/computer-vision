'''
A rotation matrix that has been correctly derived from a unit quaternion should have a few key properties:

1. Orthogonality: The rows (and columns) of the matrix should be orthogonal to each other. This means that the dot product of any two different rows or any two different columns should be zero.

2. Normalized Rows and Columns: Each row and each column of the matrix should be normalized, meaning their lengths should be equal to 1. This is because they represent the axes of the rotated coordinate system.

3. Determinant: The determinant of the rotation matrix should be +1, indicating that it represents a proper rotation without reflection.
'''
import numpy as np

# Given rotation matrix from the uploaded image
rotation_matrix = np.array([
    [-0.01227799, 0.99973782, -0.01935904],
    [-0.99976955, -0.01188216, 0.01787922],
    [0.0176445,  0.0195732,  0.99965272]
])

# Check for orthogonality by dot product of different rows and columns
orthogonal_rows = np.allclose(np.dot(rotation_matrix[0], rotation_matrix[1]), 0, atol=1e-6) and \
                  np.allclose(np.dot(rotation_matrix[0], rotation_matrix[2]), 0, atol=1e-6) and \
                  np.allclose(np.dot(rotation_matrix[1], rotation_matrix[2]), 0, atol=1e-6)

# Check for normalized rows and columns
normalized_rows = np.allclose(np.linalg.norm(rotation_matrix, axis=1), 1, atol=1e-6)
normalized_cols = np.allclose(np.linalg.norm(rotation_matrix, axis=0), 1, atol=1e-6)

# Check for determinant to be +1
det = np.linalg.det(rotation_matrix)
correct_determinant = np.allclose(det, 1, atol=1e-6)

# Printing out the results
print(orthogonal_rows, normalized_rows, normalized_cols, correct_determinant, det)
# (False, True, True, True, 1.0000006160946953)
