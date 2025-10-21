import cv2
import numpy as np


def detect_aruco_and_edges(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect ArUco
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    # all black ArUco
    mask = np.zeros_like(frame, dtype=np.uint8)

    if ids is not None:
        for corner in corners:
            # green
            pts = np.int32(corner[0])  # integer
            cv2.fillPoly(mask, [pts], (0, 255, 0))

    # noise reduction (Gaussian)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    frame_with_mask = cv2.addWeighted(frame, 1, mask, 1, 0)

    # combine ArUco and Canny
    combined_edges = cv2.bitwise_or(edges, cv2.cvtColor(
        mask[:, :, 1], cv2.COLOR_GRAY2BGR)[:, :, 1])

    return frame_with_mask, combined_edges


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("cannot open the camera")
        return

    # ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    while True:

        ret, frame = cap.read()
        if not ret:
            print("cannot read frame from camera")
            break

        # ArUco marking and Canny edge
        frame_with_mask, combined_edges = detect_aruco_and_edges(
            frame, aruco_dict, aruco_params)

        # original result
        cv2.imshow('Original with ArUco Mask', frame_with_mask)
        cv2.imshow('Canny Edges with ArUco', combined_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
