import cv2
import numpy as np


def CalculateMatrix():
    # Select ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # Create ArUco parameters
    parameters = cv2.aruco.DetectorParameters()
    avg_matrix = None

    # Open camera
    video_capture = cv2.VideoCapture(0)

    # To store image coordinates of ArUco markers
    aruco_centers_dict = {}

    # List to store homography matrices for averaging
    homography_matrices = []
    max_frames = 10  # Number of frames to calculate the average
    homography_calculated = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Cannot read frames")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        # If markers are detected
        if len(corners) == 4:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                if ids[i][0] > 3:  # Only consider markers with IDs 0-3
                    continue
                else:
                    # Calculate the center point of the ArUco marker
                    moment = cv2.moments(corners[i][0])
                    if moment["m00"] != 0:
                        cX = int(moment["m10"] / moment["m00"])
                        cY = int(moment["m01"] / moment["m00"])
                        aruco_centers_dict[ids[i][0]] = (cX, cY)

        # Get sorted ArUco marker centers
        aruco_centers = [aruco_centers_dict[i]
                         for i in sorted(aruco_centers_dict)]

        # Display detected markers
        cv2.imshow("Detected ArUco Markers", frame)

        # Only calculate homography if all required markers are detected
        if len(aruco_centers) == 4:
            pixel_coords = np.array(aruco_centers, dtype=np.float32)

            # Define real-world coordinates of the ArUco markers
            # This is an example
            # warning!!!!!!!!!!!!!!!!
            # you need to reset the real coordinate of aruco marker
            real_coords_big = np.array(
                [[275, 165, 0], [275, 515, 0], [-275, 515, 0], [-275, 165, 0]],
                dtype=np.float32
            )

            # Calculate the 3x3 transformation matrix
            proj, _ = cv2.findHomography(pixel_coords, real_coords_big)
            if proj is not None:
                homography_matrices.append(proj)

            # Stop collecting matrices after reaching the desired number of frames
            if len(homography_matrices) >= max_frames:
                homography_calculated = True

        # If sufficient frames have been captured, calculate the average matrix
        if homography_calculated:
            avg_matrix = np.mean(homography_matrices, axis=0)
            # print("Averaged Homography Matrix:")
            # print(avg_matrix)
            homography_calculated = False  # Prevent further calculation
            return avg_matrix

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
