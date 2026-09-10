"""
Eye Tracking and Head Pose Estimation

This script is designed to perform real-time eye tracking and head pose estimation using a webcam feed. 
It utilizes the MediaPipe library for facial landmark detection, which informs both eye tracking and 
head pose calculations. The purpose is to track the user's eye movements and head orientation, 
which can be applied in various domains such as HCI (Human-Computer Interaction), gaming, and accessibility tools.

Features:
- Real-time eye tracking to count blinks and calculate the eye aspect ratio for each frame.
- Head pose estimation to determine the orientation of the user's head in terms of pitch, yaw, and roll angles.
- Calibration feature to set the initial head pose as the reference zero position.
- Data logging for further analysis and debugging.

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Methodology:
- The script uses the 468 facial landmarks provided by MediaPipe's FaceMesh model.
- Eye tracking is achieved by calculating the Eye Aspect Ratio (EAR) for each eye and detecting blinks based on EAR thresholds.
- Head pose is estimated using the solvePnP algorithm with a predefined 3D facial model and corresponding 2D landmarks detected from the camera feed.
- Angles are normalized to intuitive ranges (pitch: [-90, 90], yaw and roll: [-180, 180]).

Theory:
- EAR is used as a simple yet effective metric for eye closure detection.
- Head pose angles are derived using a perspective-n-point approach, which estimates an object's pose from its 2D image points and 3D model points.

UDP Packet Structure:
- The UDP packet consists of a timestamp and four other integer values.
- Packet Type: Mixed (int64 for timestamp, int32 for other values)
- Packet Structure: [timestamp (int64), l_cx (int32), l_cy (int32), l_dx (int32), l_dy (int32)]
- Packet Size: 24 bytes (8 bytes for int64 timestamp, 4 bytes each for the four int32 values)

Example Packets:
- Example 1: [1623447890123, 315, 225, 66, -3]
- Example 2: [1623447891123, 227, 68, -1, 316]

Parameters:
You can change parameters such as face width, moving average window, webcam ID, terminal outputs, on-screen data, logging detail, etc., from the code.

Author: Alireza Bagheri
GitHub: https://github.com/alireza787b/Python-Gaze-Face-Tracker
Email: p30planets@gmail.com
LinkedIn: https://www.linkedin.com/in/alireza787b
Date: November 2023

Inspiration:
Initially inspired by Asadullah Dal's iris segmentation project (https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python). 
The blink detection feature is also contributed by Asadullah Dal (GitHub: Asadullah-Dal17).

Usage:
- Run the script in a Python environment with the necessary dependencies installed. The script accepts command-line arguments for camera source configuration.
- The script automatically runs a 9-point screen calibration at startup. Press 'c' to repeat the full screen calibration.
- Press 'r' to start/stop logging.
- Press 'q' to exit the program.
- Output is displayed in a window with live feed and annotations, and logged to a CSV file for further analysis.

Ensure that all dependencies, especially MediaPipe, OpenCV, and NumPy, are installed before running the script.

Note:
This project is intended for educational and research purposes in fields like aviation, human-computer interaction, and more.
"""


import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
from AngleBuffer import AngleBuffer


#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

# Parameters Documentation

## User-Specific Measurements
# USER_FACE_WIDTH: The horizontal distance between the outer edges of the user's cheekbones in millimeters. 
# This measurement is used to scale the 3D model points for head pose estimation.
# Measure your face width and adjust the value accordingly.
USER_FACE_WIDTH = 140  # [mm]

## Camera Parameters (not currently used in calculations)
# NOSE_TO_CAMERA_DISTANCE: The distance from the tip of the nose to the camera lens in millimeters.
# Intended for future use where accurate physical distance measurements may be necessary.
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]

## Configuration Parameters
# PRINT_DATA: Enable or disable the printing of data to the console for debugging.
PRINT_DATA = True

# DEFAULT_WEBCAM: Default camera source index. '0' usually refers to the built-in webcam.
DEFAULT_WEBCAM = 0

# SHOW_ALL_FEATURES: If True, display all facial landmarks on the video feed.
SHOW_ALL_FEATURES = True

# LOG_DATA: Enable or disable logging of data to a CSV file.
LOG_DATA = True

# LOG_ALL_FEATURES: If True, log all facial landmarks to the CSV file.
LOG_ALL_FEATURES = False

# ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
ENABLE_HEAD_POSE = True

## Logging Configuration
# LOG_FOLDER: Directory where log files will be stored.
LOG_FOLDER = "logs"

## Server Configuration
# SERVER_IP: IP address of the server for sending data via UDP (default is localhost).
SERVER_IP = "127.0.0.1"

# SERVER_PORT: Port number for the server to listen on.
SERVER_PORT = 7070

## Blink Detection Parameters
# SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.
SHOW_ON_SCREEN_DATA = True

# TOTAL_BLINKS: Counter for the total number of blinks detected.
TOTAL_BLINKS = 0

# EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
EYES_BLINK_FRAME_COUNTER = 0

# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51

# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 0  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True
LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070  # Set the server port

# eyes blinking variables
SHOW_BLINK_COUNT_ON_SCREEN = True  # Toggle to show the blink count on the video feed
TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
EYES_BLINK_FRAME_COUNTER = (
    0  # Counts the number of consecutive frames with a potential blink
)
BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
EYE_AR_CONSEC_FRAMES = (
    2  # Number of consecutive frames below the threshold to confirm a blink
)
# SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)


#If set to false it will wait for your command (hittig 'r') to start logging.
IS_RECORDING = False  # Controls whether data is being logged
RECORDING_START_TIME = None
RECORDING_SESSION_NUMBER = 0


# ============================================================================
# SCREEN CALIBRATION SETTINGS
# ============================================================================
# Calibration is performed EVERY TIME the program starts. The calibration
# targets are displayed on the COMPUTER SCREEN. The DroidCam phone is only
# the camera/sensor used to measure eye and head features.
CALIBRATION_ENABLED = True
CALIBRATION_POINT_DURATION = 1.25      # seconds to collect at each point
CALIBRATION_SETTLE_TIME = 0.75         # seconds before collection starts
CALIBRATION_MIN_SAMPLES = 20            # minimum valid frames per point
CALIBRATION_USE_HEAD_POSE = True
CALIBRATION_WINDOW_NAME = "Gaze Calibration - Look at the Dot"

# NEW: Since the camera (DroidCam on a phone) gets repositioned between
# sessions and sometimes mid-session, calibration quality varies a lot run
# to run. This threshold flags a calibration as "poor" so you can redo it
# immediately instead of tracking with an unreliable mapping. Tune this to
# your screen size / accuracy needs - it's just a heads-up, not a hard stop.
CALIBRATION_RMSE_WARNING_PIXELS = 150

# NEW: thresholds for the face_looks intent classifier, in degrees of
# deviation from the CALIBRATED neutral head pose (not raw camera-frame
# angles - see the fix at the face_looks classification site for why that
# distinction matters on an extreme/repositionable camera mount).
HEAD_YAW_THRESHOLD = 12
HEAD_PITCH_THRESHOLD = 12

# NEW: sign-flip toggles for the fixed head-pose math. Whether yaw/pitch/roll
# come out mirrored depends on the exact geometry of how the phone is
# mounted relative to the face, which isn't something that can be derived
# from code alone - test empirically:
#   - Look LEFT. If the "Face Looking at" text (or Screen X) says "Right",
#     flip INVERT_YAW.
#   - Look DOWN. If it says "Up" (or the on-screen dot moves the wrong
#     way), flip INVERT_PITCH.
#   - If tilting your head sideways (ear toward shoulder) shows the wrong
#     roll sign in the printed data, flip INVERT_ROLL.
# Re-run calibration after changing any of these.
INVERT_YAW = False
INVERT_PITCH = False
INVERT_ROLL = False
TRACKING_WINDOW_NAME = "Eye Tracking"

# Nine screen targets. Center is collected first so it can be used as the
# neutral head-pose reference for this particular phone position.
CALIBRATION_POINTS = [
    ("CENTER", 0.50, 0.50),
    ("TOP_LEFT", 0.08, 0.08),
    ("TOP_CENTER", 0.50, 0.08),
    ("TOP_RIGHT", 0.92, 0.08),
    ("MIDDLE_LEFT", 0.08, 0.50),
    ("MIDDLE_RIGHT", 0.92, 0.50),
    ("BOTTOM_LEFT", 0.08, 0.92),
    ("BOTTOM_CENTER", 0.50, 0.92),
    ("BOTTOM_RIGHT", 0.92, 0.92),
]

# The linear model uses six inputs and therefore needs at least six target
# positions. Nine points gives us redundancy while remaining fast enough for
# every run. Inputs are normalized eye position plus head pose.
calibration_model_x = None
calibration_model_y = None
calibration_screen_width = 0
calibration_screen_height = 0
calibration_rmse = None
calibration_point_data = []

# NEW: safe defaults. Previously these were only ever assigned inside the
# "if LOG_DATA and IS_RECORDING:" block, so on any frame where recording
# hadn't been started yet (e.g. right after startup calibration), the
# on-screen-display code below tried to read them before they existed at
# all, raising: NameError: name 'calibrated_screen_x' is not defined
calibrated_screen_x = None
calibrated_screen_y = None
calibrated_gaze_direction = ""


def get_screen_size():
    """Return the primary computer display resolution."""
    try:
        import ctypes
        return int(ctypes.windll.user32.GetSystemMetrics(0)), int(ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return int(w), int(h)
        except Exception:
            return 1920, 1080


def get_calibration_features(results, img_w, img_h, head_baseline=None):
    """Extract the same eye/pupil/head information used by the tracker."""
    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]
    mesh_points = np.array(
        [
            np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
            for p in face.landmark
        ]
    )

    try:
        mesh_points_3D = np.array([[n.x, n.y, n.z] for n in face.landmark])

        (l_cx, l_cy), _ = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
        (r_cx, r_cy), _ = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])

        left_eye_width = abs(
            float(mesh_points[LEFT_EYE_INNER_CORNER][0][0])
            - float(mesh_points[LEFT_EYE_OUTER_CORNER][0][0])
        )
        left_eye_height = abs(float(mesh_points[159][1]) - float(mesh_points[145][1]))
        right_eye_width = abs(
            float(mesh_points[RIGHT_EYE_INNER_CORNER][0][0])
            - float(mesh_points[RIGHT_EYE_OUTER_CORNER][0][0])
        )
        right_eye_height = abs(float(mesh_points[386][1]) - float(mesh_points[374][1]))

        if left_eye_width <= 0 or left_eye_height <= 0 or right_eye_width <= 0 or right_eye_height <= 0:
            return None

        left_horizontal = (
            float(l_cx) - float(mesh_points[LEFT_EYE_OUTER_CORNER][0][0])
        ) / left_eye_width
        left_vertical = (
            float(l_cy) - float(mesh_points[159][1])
        ) / left_eye_height
        right_horizontal = (
            float(r_cx) - float(mesh_points[RIGHT_EYE_OUTER_CORNER][0][0])
        ) / right_eye_width
        right_vertical = (
            float(r_cy) - float(mesh_points[386][1])
        ) / right_eye_height

        avg_horizontal = (left_horizontal + right_horizontal) / 2.0
        avg_vertical = (left_vertical + right_vertical) / 2.0

        # Reject closed-eye frames during calibration.
        ear = blinking_ratio(mesh_points_3D)
        if ear <= BLINK_THRESHOLD:
            return None

        pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
        if head_baseline is not None:
            pitch -= head_baseline[0]
            yaw -= head_baseline[1]
            roll -= head_baseline[2]

        return np.array(
            [avg_horizontal, avg_vertical, float(pitch), float(yaw), float(roll)],
            dtype=np.float64,
        )
    except Exception:
        return None


def make_calibration_design_matrix(features):
    """Create [1, eye-x, eye-y, pitch, yaw, roll] model inputs."""
    return np.array(
        [[1.0, f[0], f[1], f[2], f[3], f[4]] for f in features],
        dtype=np.float64,
    )


def draw_calibration_screen(window_name, screen_w, screen_h, point_name, px, py,
                            instruction, progress_text, countdown=None):
    """Draw one calibration target in a fullscreen OpenCV window."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv.putText(canvas, instruction, (50, 60), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(canvas, progress_text, (50, 105), cv.FONT_HERSHEY_SIMPLEX,
               0.8, (200, 200, 200), 2, cv.LINE_AA)

    cv.circle(canvas, (px, py), 18, (255, 255, 255), -1, cv.LINE_AA)
    cv.circle(canvas, (px, py), 7, (0, 0, 0), -1, cv.LINE_AA)

    if countdown is not None:
        cv.putText(canvas, str(countdown), (screen_w // 2 - 15, screen_h // 2),
                   cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv.LINE_AA)

    cv.imshow(window_name, canvas)


def run_screen_calibration(cap, face_mesh):
    """Run a fresh 9-point calibration against the computer screen."""
    global initial_pitch, initial_yaw, initial_roll, calibrated
    global calibration_model_x, calibration_model_y
    global calibration_screen_width, calibration_screen_height, calibration_rmse
    global calibration_point_data

    calibration_screen_width, calibration_screen_height = get_screen_size()
    calibration_point_data = []

    cv.namedWindow(CALIBRATION_WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.setWindowProperty(CALIBRATION_WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    print("\n============================================================")
    print("SCREEN CALIBRATION STARTED")
    print("The dots are on the COMPUTER SCREEN.")
    print("Keep your HEAD as still as possible and move only your EYES.")
    print("Follow the dot naturally. Do not look at the DroidCam preview.")
    print("Press Q during calibration to cancel.")
    print("============================================================\n")

    # Center is deliberately first: it establishes the neutral head pose for
    # this exact phone position before the other eight targets are measured.
    raw_features_by_point = []

    for point_index, (point_name, nx, ny) in enumerate(CALIBRATION_POINTS):
        px = int(nx * (calibration_screen_width - 1))
        py = int(ny * (calibration_screen_height - 1))

        # Settle/countdown period.
        settle_start = time.time()
        while time.time() - settle_start < CALIBRATION_SETTLE_TIME:
            ret, frame = cap.read()
            if not ret:
                continue
            draw_calibration_screen(
                CALIBRATION_WINDOW_NAME,
                calibration_screen_width,
                calibration_screen_height,
                point_name, px, py,
                "Look at the dot and keep your head still",
                f"Calibration point {point_index + 1} / {len(CALIBRATION_POINTS)}: {point_name}",
                countdown=max(1, int(CALIBRATION_SETTLE_TIME - (time.time() - settle_start)) + 1),
            )
            key = cv.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                cv.destroyWindow(CALIBRATION_WINDOW_NAME)
                raise KeyboardInterrupt("Calibration cancelled by user.")

        # For the center point, estimate the neutral pose from its samples.
        samples = []
        collection_start = time.time()
        while time.time() - collection_start < CALIBRATION_POINT_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            results = face_mesh.process(rgb)

            baseline_for_features = None if point_name == "CENTER" else (initial_pitch, initial_yaw, initial_roll)
            features = get_calibration_features(results, w, h, baseline_for_features)
            if features is not None:
                samples.append(features)

            elapsed = time.time() - collection_start
            progress = min(1.0, elapsed / CALIBRATION_POINT_DURATION)
            bar_w = int((calibration_screen_width - 100) * progress)
            canvas = np.zeros((calibration_screen_height, calibration_screen_width, 3), dtype=np.uint8)
            cv.putText(canvas, "Look at the dot and keep your head still", (50, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(canvas, f"Calibration point {point_index + 1} / {len(CALIBRATION_POINTS)}: {point_name}",
                       (50, 105), cv.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv.LINE_AA)
            cv.circle(canvas, (px, py), 18, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(canvas, (px, py), 7, (0, 0, 0), -1, cv.LINE_AA)
            cv.rectangle(canvas, (50, calibration_screen_height - 60),
                         (50 + bar_w, calibration_screen_height - 35),
                         (255, 255, 255), -1)
            cv.putText(canvas, f"Samples: {len(samples)}", (50, calibration_screen_height - 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv.LINE_AA)
            cv.imshow(CALIBRATION_WINDOW_NAME, canvas)

            key = cv.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                cv.destroyWindow(CALIBRATION_WINDOW_NAME)
                raise KeyboardInterrupt("Calibration cancelled by user.")

        if len(samples) < CALIBRATION_MIN_SAMPLES:
            print(f"Calibration failed at {point_name}: only {len(samples)} valid samples.")
            cv.destroyWindow(CALIBRATION_WINDOW_NAME)
            return False

        mean_features = np.mean(np.vstack(samples), axis=0)

        if point_name == "CENTER":
            # Center gaze defines the neutral head orientation for this run.
            initial_pitch = float(mean_features[2])
            initial_yaw = float(mean_features[3])
            initial_roll = float(mean_features[4])
            calibrated = True
            # Recompute the stored center features relative to its own baseline.
            mean_features = np.array([
                mean_features[0], mean_features[1], 0.0, 0.0, 0.0
            ], dtype=np.float64)

        calibration_point_data.append({
            "name": point_name,
            "target_x": float(px),
            "target_y": float(py),
            "features": mean_features.copy(),
            "sample_count": len(samples),
        })
        raw_features_by_point.append(mean_features.copy())

        print(f"  {point_name:14s}: {len(samples):3d} samples")

    cv.destroyWindow(CALIBRATION_WINDOW_NAME)

    features = [p["features"] for p in calibration_point_data]
    X = make_calibration_design_matrix(features)
    target_x = np.array([p["target_x"] for p in calibration_point_data], dtype=np.float64)
    target_y = np.array([p["target_y"] for p in calibration_point_data], dtype=np.float64)

    # Least-squares linear mapping. This is intentionally lightweight so it is
    # re-fitted quickly every time the phone moves.
    calibration_model_x, _, _, _ = np.linalg.lstsq(X, target_x, rcond=None)
    calibration_model_y, _, _, _ = np.linalg.lstsq(X, target_y, rcond=None)

    pred_x = X @ calibration_model_x
    pred_y = X @ calibration_model_y
    errors = np.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
    calibration_rmse = float(np.sqrt(np.mean(errors ** 2)))

    print("\n============================================================")
    print("SCREEN CALIBRATION COMPLETE")
    print(f"Screen resolution : {calibration_screen_width} x {calibration_screen_height}")
    print(f"Calibration RMSE  : {calibration_rmse:.1f} pixels")
    print(f"Neutral Pitch     : {initial_pitch:.3f}")
    print(f"Neutral Yaw       : {initial_yaw:.3f}")
    print(f"Neutral Roll      : {initial_roll:.3f}")
    print("============================================================\n")

    return True


def predict_screen_gaze(avg_horizontal, avg_vertical, pitch, yaw, roll):
    """Convert current eye/head features to calibrated screen coordinates."""
    if calibration_model_x is None or calibration_model_y is None:
        return None, None

    # NOTE: this used to build `feature` as a (1, 6) 2D array, so
    # `feature @ calibration_model_x` produced a (1,) array rather than a
    # true scalar. float() on that raised, under NumPy 2.x:
    #   "only 0-dimensional arrays can be converted to Python scalars"
    # Using a 1D (6,) feature vector makes the dot product a real scalar.
    feature = np.array(
        [1.0, avg_horizontal, avg_vertical, pitch, yaw, roll], dtype=np.float64
    )
    x = float(feature @ calibration_model_x)
    y = float(feature @ calibration_model_y)
    x = max(0.0, min(float(calibration_screen_width - 1), x))
    y = max(0.0, min(float(calibration_screen_height - 1), y))
    return x, y


def screen_gaze_direction(x, y, width, height):
    """Return a 3x3 screen direction label from calibrated X/Y."""
    if x is None or y is None or width <= 0 or height <= 0:
        return "Unknown"

    h = "Left" if x < width * 0.33 else "Right" if x > width * 0.67 else "Center"
    v = "Up" if y < height * 0.33 else "Down" if y > height * 0.67 else "Center"

    if h == "Center" and v == "Center":
        return "Center"
    if h == "Center":
        return v
    if v == "Center":
        return h
    return v + "-" + h

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]  # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

# Server address for UDP socket communication
SERVER_ADDRESS = (SERVER_IP, 7070)


# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1


def euclidean_distance_3D(points):
    """Calculates the Euclidean distance between two points in 3D space.

    Args:
        points: A list of 3D points.

    Returns:
        The Euclidean distance between the two points.

        # Comment: This function calculates the Euclidean distance between two points in 3D space.
    """

    # Get the three points.
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    # Calculate the numerator.
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    # Calculate the denominator.
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    # Calculate the distance.
    distance = numerator / denominator

    return distance

def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = USER_FACE_WIDTH / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])
    

    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")


        # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # ------------------------------------------------------------------
    # FIX: extreme camera mounting angle broke pitch (forward/backward).
    #
    # The old code went through cv.decomposeProjectionMatrix() and then a
    # hand-tuned normalize_pitch() with several branch-based sign flips.
    # That combination is only well-behaved for a camera mounted roughly
    # level and centered on the face - it can SATURATE (multiple different
    # real head tilts map to the same "forward" output) or FOLD
    # (continuing to tilt further reverses the direction of change) once
    # the underlying raw angle moves outside that narrow assumed range.
    # That's exactly the "always thinks forward" / "descends but not the
    # usual way" behavior you saw with the phone mounted lower.
    #
    # rotation_matrix_to_euler_angles() below uses a standard atan2-based
    # decomposition instead, which is continuous and monotonic over the
    # full range relevant here (only singular exactly at +/-90 degrees,
    # which a face pointed roughly at the camera won't hit). No folding,
    # no saturation, regardless of mounting angle.
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    # FIX: lateral (left/right) inversion.
    #
    # The model's 3D axis convention (Y up, Z toward the viewer) doesn't
    # match OpenCV's camera-frame convention (Y down, Z into the scene),
    # so the raw sign of yaw/pitch/roll coming out of any Euler
    # decomposition depends on exactly how solvePnP resolved that
    # mismatch - it can come out mirrored. Rather than guess, this is
    # exposed as toggles you can flip based on what you observe:
    if INVERT_YAW:
        yaw = -yaw
    if INVERT_PITCH:
        pitch = -pitch
    if INVERT_ROLL:
        roll = -roll
    # ------------------------------------------------------------------

    return pitch, yaw, roll


def rotation_matrix_to_euler_angles(R):
    """Standard atan2-based rotation-matrix -> Euler angle decomposition
    (pitch, yaw, roll, in degrees). Continuous and monotonic, unlike
    cv.decomposeProjectionMatrix's output - see the usage site for why
    that distinction matters at extreme camera mounting angles.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


# NEW: kept only for reference / in case anything external imports it.
# No longer called anywhere in this script - see the fix notes above
# estimate_head_pose() for why it was replaced.
def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch


# This function calculates the blinking ratio of a person.
def blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person.

    Args:
        landmarks: A facial landmarks in 3D normalized.

    Returns:
        The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

    """

    # Get the right eye ratio.
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

    # Get the left eye ratio.
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

    # Calculate the blinking ratio.
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio


# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    if PRINT_DATA:
        head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
        print(f"Head pose estimation is {head_pose_status}.")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Preparing for CSV logging
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Column names for CSV file
column_names = [
    "Timestamp (ms)",
    "Date Time",
    "Recording Session",
    "Recording Elapsed Time (s)",
    "Frame Number",

    "Left Eye Center X",
    "Left Eye Center Y",
    "Left Eye Width",
    "Left Eye Height",
    "Left Pupil X",
    "Left Pupil Y",
    "Left Pupil Horizontal",
    "Left Pupil Vertical",

    "Right Eye Center X",
    "Right Eye Center Y",
    "Right Eye Width",
    "Right Eye Height",
    "Right Pupil X",
    "Right Pupil Y",
    "Right Pupil Horizontal",
    "Right Pupil Vertical",

    "Average Pupil X",
    "Average Pupil Y",
    "Average Pupil Horizontal",
    "Average Pupil Vertical",

    "Left Eye Aspect Ratio",
    "Right Eye Aspect Ratio",
    "Average Eye Aspect Ratio",

    "Blink",
    "Total Blink Count",

    "Gaze Horizontal",
    "Gaze Vertical",
    "Gaze Direction",

    "Pitch",
    "Yaw",
    "Roll",

    "Calibrated Screen X",
    "Calibrated Screen Y",
    "Calibrated Gaze Direction",
    "Calibration RMSE (pixels)"
]
if LOG_ALL_FEATURES:
    column_names.extend(
        [f"Landmark_{i}_X" for i in range(468)]
        + [f"Landmark_{i}_Y" for i in range(468)]
    )

# Run a fresh screen calibration for every program launch.
# This is deliberately BEFORE normal tracking so a moved DroidCam position
# gets its own mapping.
#
# IMPORTANT (phone/DroidCam setup): calibration is always done against the
# COMPUTER SCREEN, never against the phone itself. The phone is just the
# sensor - wherever it happens to be pointed this session, the 9-point
# calibration learns the mapping from "what the camera currently sees" to
# "where that corresponds to on the screen," and also resets the neutral
# head-pose baseline to the phone's current angle. That's exactly why a
# fresh calibration is required every time the phone moves: the mapping
# and the neutral pose are both tied to the phone's CURRENT position, not
# to some fixed absolute reference.
if CALIBRATION_ENABLED:
    try:
        while True:
            if not run_screen_calibration(cap, mp_face_mesh):
                raise RuntimeError("Screen calibration failed. Please restart and ensure your face is visible.")

            # NEW: catch a bad calibration (common right after repositioning
            # the phone) before you start tracking/recording with it.
            if calibration_rmse is not None and calibration_rmse > CALIBRATION_RMSE_WARNING_PIXELS:
                print(f"WARNING: calibration RMSE is {calibration_rmse:.1f}px, which is high.")
                print("This usually means the phone moved during calibration, the face")
                print("wasn't clearly visible, or the head pose drifted between points.")
                choice = input("Press ENTER to continue anyway, or type R + ENTER to redo calibration now: ").strip().lower()
                if choice == "r":
                    continue
            break
    except KeyboardInterrupt:
        print("Calibration cancelled. Exiting program.")
        cap.release()
        cv.destroyAllWindows()
        raise SystemExit

# Main loop for video capture and processing
frame_number = 0
try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Flipping the frame for a mirror effect
        # I think we better not flip to correspond with real world... need to make sure later...
        #frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            # Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
            # just normalize values
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            # getting the head pose estimation 3d points
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

            # collect nose three dimension and two dimension points
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            # Get rotational matrix
            rotation_matrix, jac = cv.Rodrigues(rot_vec)

            # Get angles (RAW, camera-frame-relative - NOT corrected for how
            # the phone is mounted). Only used below for the cosmetic pink
            # nose-direction line. Do NOT use angle_x/angle_y for intent -
            # see the fix below for why.
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            # ------------------------------------------------------------
            # FIX: intent under an extreme/repositionable camera mount.
            #
            # The old code classified face_looks ("Left"/"Right"/"Up"/"Down")
            # straight from angle_x/angle_y above - the RAW head pose in the
            # camera's own frame, compared against a fixed +/-10 degree
            # threshold. That's only correct if the camera sits level and
            # dead-center in front of the face. On a wheelchair-mounted
            # phone at an extreme angle (like your screenshot), the raw
            # angles sit at some large non-zero baseline even when the user
            # is looking straight ahead relative to their own body/chair -
            # so this classifier reported "Left"/"Up" almost constantly,
            # regardless of actual intent.
            #
            # The fix: compute head pose the same way the screen-gaze model
            # does - via estimate_head_pose() - and subtract the NEUTRAL
            # pose captured during calibration's CENTER point. That neutral
            # pose already encodes wherever the phone happens to be mounted
            # THIS session. The result is a pitch/yaw relative to "however
            # the user's head was oriented when they looked straight ahead
            # during calibration," which stays meaningful no matter how
            # extreme or off-axis the physical camera position is.
            #
            # This must run before the face_looks classification below, and
            # before the calibrated-gaze block further down reuses the same
            # pitch/yaw/roll values.
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()

                if calibrated:
                    pitch -= initial_pitch
                    yaw -= initial_yaw
                    roll -= initial_roll
            else:
                pitch, yaw, roll = 0.0, 0.0, 0.0

            # Classify intent from the CALIBRATED relative pose, not the raw
            # camera-frame angles. Thresholds are in degrees of deviation
            # from the calibrated neutral pose.
            if yaw < -HEAD_YAW_THRESHOLD:
                face_looks = "Left"
            elif yaw > HEAD_YAW_THRESHOLD:
                face_looks = "Right"
            elif pitch < -HEAD_PITCH_THRESHOLD:
                face_looks = "Down"
            elif pitch > HEAD_PITCH_THRESHOLD:
                face_looks = "Up"
            else:
                face_looks = "Forward"
            # ------------------------------------------------------------

            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            # Display the nose direction (cosmetic only - still uses the raw
            # angle_x/angle_y, so this pink line can look "off" relative to
            # the calibrated Face Looking / gaze readouts on an extreme
            # camera mount. That's expected and harmless; it's just a
            # visual debug aid, not an intent signal.
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
            # getting the blinking ratio
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            # print(f"Blinking ratio : {ratio}")
            # checking if ear less then or equal to required threshold if yes then
            # count the number of frame frame while eyes are closed.
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            # else check if eyes are closed is greater EYE_AR_CONSEC_FRAMES frame then
            # count the this as a blink
            # make frame counter equal to zero

            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            
            # Display all facial landmarks if enabled
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            cv.circle(
                frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Left eye right corner
            cv.circle(
                frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Left eye left corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Right eye right corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Total Blinks: {TOTAL_BLINKS}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
                if ENABLE_HEAD_POSE:
                    print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")

            # NEW: this whole block used to live inside
            # "if LOG_DATA and IS_RECORDING:", which meant calibrated_screen_x/
            # calibrated_screen_y/calibrated_gaze_direction were only ever
            # computed while a recording session was active. But the on-screen
            # display code further below reads those same variables every
            # frame regardless of recording state, which crashed with
            # NameError as soon as calibration finished (before you pressed
            # 'r'). Now the eye-metric / screen-gaze calculation always runs
            # when a face is detected, and only the CSV row is gated on
            # recording being on. (No "global" keyword needed here - this
            # loop runs at module scope, not inside a function.)

            # Eye dimensions from the existing MediaPipe landmarks.
            # NOTE: this now runs every frame a face is detected (not only
            # while recording), because calibrated_screen_x/y and
            # calibrated_gaze_direction are needed for the live on-screen
            # display below regardless of recording state.
            left_eye_width = abs(
                mesh_points[LEFT_EYE_INNER_CORNER][0][0]
                - mesh_points[LEFT_EYE_OUTER_CORNER][0][0]
            )
            left_eye_height = abs(
                mesh_points[159][1] - mesh_points[145][1]
            )

            right_eye_width = abs(
                mesh_points[RIGHT_EYE_INNER_CORNER][0][0]
                - mesh_points[RIGHT_EYE_OUTER_CORNER][0][0]
            )
            right_eye_height = abs(
                mesh_points[386][1] - mesh_points[374][1]
            )

            # Existing iris centers.
            left_pupil_x = float(l_cx)
            left_pupil_y = float(l_cy)
            right_pupil_x = float(r_cx)
            right_pupil_y = float(r_cy)

            # Relative pupil positions.
            left_horizontal = (
                (left_pupil_x - mesh_points[LEFT_EYE_OUTER_CORNER][0][0])
                / float(left_eye_width)
                if float(left_eye_width) != 0.0 else 0.0
            )
            left_vertical = (
                (left_pupil_y - mesh_points[159][1])
                / float(left_eye_height)
                if float(left_eye_height) != 0.0 else 0.0
            )

            right_horizontal = (
                (right_pupil_x - mesh_points[RIGHT_EYE_OUTER_CORNER][0][0])
                / float(right_eye_width)
                if float(right_eye_width) != 0.0 else 0.0
            )
            right_vertical = (
                (right_pupil_y - mesh_points[386][1])
                / float(right_eye_height)
                if float(right_eye_height) != 0.0 else 0.0
            )

            average_pupil_x = (
                left_pupil_x + right_pupil_x
            ) / 2.0
            average_pupil_y = (
                left_pupil_y + right_pupil_y
            ) / 2.0
            average_horizontal = (
                left_horizontal + right_horizontal
            ) / 2.0
            average_vertical = (
                left_vertical + right_vertical
            ) / 2.0

            # Calibrated gaze: map current camera measurements to the
            # COMPUTER SCREEN coordinates learned at startup.
            calibrated_screen_x, calibrated_screen_y = predict_screen_gaze(
                average_horizontal, average_vertical, pitch, yaw, roll
            )
            calibrated_gaze_direction = screen_gaze_direction(
                calibrated_screen_x, calibrated_screen_y,
                calibration_screen_width, calibration_screen_height
            )

            # Pupil-based gaze direction (kept for comparison with the
            # original script).
            if average_horizontal < 0.35:
                gaze_horizontal = "Left"
            elif average_horizontal > 0.65:
                gaze_horizontal = "Right"
            else:
                gaze_horizontal = "Center"

            if average_vertical < 0.35:
                gaze_vertical = "Up"
            elif average_vertical > 0.65:
                gaze_vertical = "Down"
            else:
                gaze_vertical = "Center"

            if gaze_horizontal == "Center" and gaze_vertical == "Center":
                gaze_direction = "Center"
            elif gaze_horizontal == "Center":
                gaze_direction = gaze_vertical
            elif gaze_vertical == "Center":
                gaze_direction = gaze_horizontal
            else:
                gaze_direction = gaze_vertical + "-" + gaze_horizontal

            blink = 1 if eyes_aspect_ratio <= BLINK_THRESHOLD else 0

            # Use the EAR values already calculated by the original code.
            left_ear = blinking_ratio(mesh_points_3D)
            right_ear = blinking_ratio(mesh_points_3D)

            # Logging data
            # The CSV row is only built and saved while R has started
            # recording; the metrics above are still computed every frame
            # so the live overlay always has fresh values to show.
            if LOG_DATA and IS_RECORDING:
                timestamp = int(time.time() * 1000)
                date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                recording_elapsed = (
                    time.time() - RECORDING_START_TIME
                    if RECORDING_START_TIME is not None
                    else 0.0
                )

                log_entry = [
                    timestamp,
                    date_time,
                    RECORDING_SESSION_NUMBER,
                    round(recording_elapsed, 4),
                    frame_number,

                    round(l_cx, 3),
                    round(l_cy, 3),
                    round(left_eye_width, 3),
                    round(left_eye_height, 3),
                    round(left_pupil_x, 3),
                    round(left_pupil_y, 3),
                    round(left_horizontal, 5),
                    round(left_vertical, 5),

                    round(r_cx, 3),
                    round(r_cy, 3),
                    round(right_eye_width, 3),
                    round(right_eye_height, 3),
                    round(right_pupil_x, 3),
                    round(right_pupil_y, 3),
                    round(right_horizontal, 5),
                    round(right_vertical, 5),

                    round(average_pupil_x, 3),
                    round(average_pupil_y, 3),
                    round(average_horizontal, 5),
                    round(average_vertical, 5),

                    round(left_ear, 5),
                    round(right_ear, 5),
                    round(eyes_aspect_ratio, 5),

                    blink,
                    TOTAL_BLINKS,

                    gaze_horizontal,
                    gaze_vertical,
                    gaze_direction,

                    round(pitch, 5),
                    round(yaw, 5),
                    round(roll, 5),

                    round(calibrated_screen_x, 2) if calibrated_screen_x is not None else "",
                    round(calibrated_screen_y, 2) if calibrated_screen_y is not None else "",
                    calibrated_gaze_direction,
                    round(calibration_rmse, 2) if calibration_rmse is not None else ""
                ]

                csv_data.append(log_entry)

            # UDP transmission disabled.
            # This version is only for detailed CSV recording.
            # The original UDP code was sending a packet on every frame,
            # even when R recording was OFF.

        # Writing the on screen data on the frame
            if SHOW_ON_SCREEN_DATA:
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if calibrated_screen_x is not None:
                    cv.putText(frame, f"Screen X: {int(calibrated_screen_x)} Y: {int(calibrated_screen_y)}",
                               (30, 205), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Gaze: {calibrated_gaze_direction}",
                               (30, 235), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)

                # NEW: live calibration-quality readout + hotkey reminder.
                # Useful with a handheld/repositionable phone camera, where
                # the mapping can go stale mid-session without any other
                # visible warning sign.
                if calibration_rmse is not None:
                    rmse_color = (
                        (0, 0, 255) if calibration_rmse > CALIBRATION_RMSE_WARNING_PIXELS
                        else (0, 255, 0)
                    )
                    cv.putText(frame, f"Calib RMSE: {calibration_rmse:.0f}px",
                               (30, img_h - 45), cv.FONT_HERSHEY_DUPLEX, 0.6, rmse_color, 2, cv.LINE_AA)
                cv.putText(frame, "Press C to recalibrate if the phone moved",
                           (30, img_h - 15), cv.FONT_HERSHEY_DUPLEX, 0.55, (200, 200, 200), 1, cv.LINE_AA)


        
        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Press C to perform the COMPLETE screen calibration again.
        # This is useful if the phone is moved while the program is running.
        if key == ord('c') or key == ord('C'):
            if IS_RECORDING:
                print("Stop recording with R before recalibrating.")
            else:
                print("Starting full screen recalibration...")
                if not run_screen_calibration(cap, mp_face_mesh):
                    print("Recalibration failed. Keeping the previous calibration.")
                else:
                    print("Recalibration successful.")
                    # NEW: same RMSE heads-up as the startup calibration.
                    # Doesn't block here (you can just press C again if needed).
                    if calibration_rmse is not None and calibration_rmse > CALIBRATION_RMSE_WARNING_PIXELS:
                        print(f"WARNING: RMSE is {calibration_rmse:.1f}px (high). "
                              f"Consider pressing C again if tracking looks off.")

        # Inside the main loop, handle the 'r' key press.
        if key == ord('r') or key == ord('R'):

            if not IS_RECORDING:
                IS_RECORDING = True
                RECORDING_START_TIME = time.time()
                RECORDING_SESSION_NUMBER += 1
                csv_data = []

                print()
                print("======================================")
                print("RECORDING STARTED")
                print(f"Session: {RECORDING_SESSION_NUMBER}")
                print("Press R again to STOP and save this CSV.")
                print("======================================")
                print()

            else:
                IS_RECORDING = False

                print()
                print("======================================")
                print("RECORDING STOPPED")
                print("======================================")
                print()

                if LOG_DATA and len(csv_data) > 0:
                    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    csv_file_name = os.path.join(
                        LOG_FOLDER,
                        f"eye_tracking_log_{timestamp_str}_session_{RECORDING_SESSION_NUMBER}.csv"
                    )

                    with open(
                        csv_file_name,
                        "w",
                        newline="",
                        encoding="utf-8"
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow(column_names)
                        writer.writerows(csv_data)

                    if PRINT_DATA:
                        print(f"Data written to {csv_file_name}")


        # Exit on 'q' key press
        if key == ord('q') or key == ord('Q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
        
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()
    if PRINT_DATA:
        print("Program exited successfully.")

    # Writing data to CSV if the program is exited while a recording
    # session is still active.
    if LOG_DATA and len(csv_data) > 0 and IS_RECORDING:
        if PRINT_DATA:
            print("Writing data to CSV...")

        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(
            LOG_FOLDER, f"eye_tracking_log_{timestamp_str}_session_{RECORDING_SESSION_NUMBER}.csv"
        )

        with open(csv_file_name, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(csv_data)

        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")
