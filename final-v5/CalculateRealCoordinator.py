import cv2
import numpy as np
import cv2.aruco as aruco


def process_video(homography_matrix):
    max_frames = 100  # Number of frames to calculate the average
    homography_calculated = False

    def calculate_real_coordinates(lego_center):
        lego_point = np.array(
            [lego_center[0], lego_center[1], 1], dtype=np.float32)
        lego_real_coord = np.dot(homography_matrix, lego_point)
        lego_real_coord /= lego_real_coord[2]
        return lego_real_coord

    def is_not_grey(frame, contour):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        mean_color = cv2.mean(frame, mask=mask)[:3]
        mean_color_hsv = cv2.cvtColor(
            np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
        min_saturation = 40
        min_value = 40
        if mean_color_hsv[1] > min_saturation and min_value < mean_color_hsv[2] < 220:
            return True  # Object is not black, white, or grey
        return False

    # Open video stream
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open the camera")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    MIN_AREA_THRESHOLD = 1000

    frame_count = 0
    center_3d_accumulated = np.zeros(2)  # Accumulate x and y coordinates
    angles_accumulated = 0  # Accumulate angles

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Unable to read video frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 2.0)
        edges = cv2.Canny(blurred_frame, threshold1=50, threshold2=150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect ArUco marker bounding boxes
        corners, ids, rejected = aruco.detectMarkers(
            gray_frame, aruco_dict, parameters=aruco_params)

        aruco_bounding_boxes = []
        if ids is not None:
            for corner in corners:
                aruco_bounding_boxes.append(
                    cv2.boundingRect(corner.reshape(4, 2)))

        largest_square = None
        max_area = 0
        approx_3d = []

        for contour in contours:
            if is_not_grey(frame, contour):
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4 and cv2.isContourConvex(approx):
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    area = cv2.contourArea(contour)
                    is_aruco = any(cv2.pointPolygonTest(corner.reshape(
                        4, 2), (x + w // 2, y + h // 2), False) >= 0 for corner in corners)

                    if not is_aruco and 0.5 <= aspect_ratio <= 2.0 and area > max_area and area > MIN_AREA_THRESHOLD:
                        largest_square = approx
                        max_area = area
                        approx_3d = [calculate_real_coordinates(
                            point[0]) for point in approx]

        if largest_square is not None:
            cv2.drawContours(frame, [largest_square], -1, (0, 255, 0), 3)

            # Calculate the geometric center
            M = cv2.moments(largest_square)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Convert the geometric center to world coordinates
                center_3d = calculate_real_coordinates((cx, cy))
                # print("Real coordinates:")
                # print(center_3d[0], center_3d[1])

                # Accumulate the 3D center coordinates and angles
                center_3d_accumulated += center_3d[:2]

                approx_3d = np.array(approx_3d, dtype=np.float32)

                # Calculate vectors and project onto the x-y plane
                vectors = approx_3d - center_3d
                vectors[:, 2] = 0

                # Calculate angles of each vector relative to the x-axis
                angles = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi
                sum_angle = np.sum(angles)

                # Normalize angles to [0, 360]
                while sum_angle < 0:
                    sum_angle += 360
                while sum_angle > 360:
                    sum_angle -= 360

                angles_accumulated += sum_angle / 4

                # Increment frame count
                frame_count += 1

                # When 10 frames are accumulated, calculate and display the average
                if frame_count == max_frames:
                    avg_center_3d = center_3d_accumulated / max_frames
                    avg_angle = angles_accumulated / max_frames

                    # print("Average real coordinates (x, y):", avg_center_3d)
                    # print("Average angle:", avg_angle)

                    # Reset accumulators
                    center_3d_accumulated = np.zeros(2)
                    angles_accumulated = 0
                    frame_count = 0
                    return avg_center_3d, avg_angle

        cv2.imshow('Detected Squares with Center', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
