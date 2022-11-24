import numpy as np
import math

sumAtA = np.zeros((5,5))
sumAtB = np.zeros((5,1))

PI = 3.141592653589793
cu = 640
cv = 472
focal_length_u = 425.097722
focal_length_v = 424.480319
radial_fisheye_dist = (0.895544, 0.060172, -0.009280, -0.005502, 0.000468)

k1 = radial_fisheye_dist[0]
k2 = radial_fisheye_dist[1]
k3 = radial_fisheye_dist[2]
k4 = radial_fisheye_dist[3]
k5 = radial_fisheye_dist[4]

for v in range(944):
  for u in range(1280):
    u_n = (u-cu)/focal_length_u
    v_n = (v-cv)/focal_length_v
    
    r = math.sqrt(u_n**2 + v_n**2)
    theta = math.atan2(r,1)
    theta_d = k1*theta + k2*theta**3 + k3*theta**5 + k4*theta**7 + k5*theta**9
    A = np.matrix([theta_d, theta_d**3, theta_d**5, theta_d**7, theta_d**9])
    b = np.matrix([theta])
    AtA = np.matmul(A.T, A)
    Atb = np.matmul(A.T, b)
    sumAtA += AtA
    sumAtb += Atb
    
x = np.linalg.solve(sumAtA, sumAtb)

