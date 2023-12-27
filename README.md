# Computer-Vision

The computer vision theory of

1. Camera Geometry
2. Distortion and Undistortion in Fisheye Lens Model
3. Distortion and Undistortion in Pinhole Lens Model
4. ViewMatrix Composition
```
K = [[fx  0  cx]
     [0  fy  cy]
     [0  0   1 ]]

Extrinsic = [[r11 r12 r13 t1]
             [r21 r22 r23 t2]
             [r31 r32 r33 t3]]

R = [[r11 r12 r13]
     [r21 r22 r23]
     [r31 r32 r33]]

viewMatrix_3by3 = [[fx*r11 + 0*r21 + cx*r31, fx*r12 + 0*r22 + cx*r32, fx*r13 + 0*r23 + cx*r33],
                   [0*r11 + fy*r21 + cy*r31, 0*r12 + fy*r22 + cy*r32, 0*r13 + fy*r23 + cy*r33],
                   [0*r11 + 0*r21 + 1*r31,   0*r12 + 0*r22 + 1*r32,   0*r13 + 0*r23 + 1*r33]]

viewMatrix_3by3[2] = [0*r11 + 0*r21 + 1*r31, 0*r12 + 0*r22 + 1*r32, 0*r13 + 0*r23 + 1*r33]
                   = [r31, r32, r33]

m1 = viewMatrix_3by3 @ viewMatrix_3by3.T

m1[0, 0] = (fx*r11 + cx*r31)^2 + (fx*r12 + cx*r32)^2 + (fx*r13 + cx*r33)^2
m1[2, 2] = [r31, r32, r33] . [r31, r32, r33]
         = r31^2 + r32^2 + r33^2
         = 1

m2[0, 0] = fx^2 * (r11^2 + r12^2 + r13^2) + 2*fx*cx * (r11*r31 + r12*r32 + r13*r33) + cx^2 * (r31^2 + r32^2 + r33^2) = fx^2 + cx^2

```

### Further Information : https://seulgi-kim.tistory.com/category/Computer%20Vision
