## CalculateMatrix.py

Output Transformation Matrix according to four ArUco points as input

```bash
if homography_calculated:
            avg_matrix = np.mean(homography_matrices, axis=0)
            print("Averaged Homography Matrix:")
            print(avg_matrix)
            homography_calculated = False  # Prevent further calculation
            return avg_matrix

```

## CalculateRealCoordinator.py

- Calculate real coordinator of Lego brick according to the its image coordinator and Transformation Matrix
- And then Calculate angle of Lego brick

```bash
center_3d = calculate_real_coordinates((cx, cy))
print("real coordinator")
print(center_3d[0],center_3d[1])
approx_3d = np.array(approx_3d, dtype=np.float32)
# Calculate vectors and project onto the x-y plane
vectors = approx_3d - center_3d
vectors[:, 2] = 0

```

## Implementation

- step 1

Modify code in CalculateMatrix.py to fit your actual real-world coordinators of the ArUco markers

```bash
# Define real-world coordinators of the ArUco markers
# This is an example
real_coords_big = np.array(
            [[230, -125, 0], [230, 108, 0], [459, -125, 0], [459, 108, 0]],
            dtype=np.float32
)
```

- step 2

There are steps to run CalculateMatrix.py first and then run CalculateRealCoordinator.py

```bash
python main.py
```
