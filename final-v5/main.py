from CalculateMatrix import CalculateMatrix
from CalculateRealCoordinator import process_video
from InverseKinematic import inverse_kinematic_grasp

# warning!!!!!!!!!!!!!!!!
# you need to reset the real coordinate of aruco marker


def main():
    # Step 1: Calculate the homography matrix
    print("································································")
    print("Calculating the homography matrix...")
    print("································································")

    homography_matrix = CalculateMatrix()

    if homography_matrix is None:
        print("································································")
        print("Failed to calculate the homography matrix.")
        print("································································")
        return

    print("································································")
    print("Homography matrix calculated successfully:")
    print("································································")
    print(homography_matrix)

    # Step 2: Use the matrix in real coordinate calculations
    print("································································")
    print("Starting real coordinate calculation...")
    print("································································")
    real_coordinator, real_angle = process_video(homography_matrix)
    x_real = int(real_coordinator[0])
    y_real = int(real_coordinator[1])
    if x_real > 0:
        x_real += 20
    
    if x_real < 0:
        x_real -=10

    print(x_real, y_real, real_angle)

    print("································································")

    #inverse_kinematic_grasp(x_real, y_real, 90 - real_angle)


if __name__ == "__main__":
    main()
