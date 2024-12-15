import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define the acceptable angle ranges for the Goddess Pose
CORRECT_ANGLES = {
    "left_elbow": (150, 180),
    "right_elbow": (150, 180),
    "left_knee": (80, 100),
    "right_knee": (80, 100),
}

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle ABC (in degrees) given three points A, B, and C.
    A, B, and C should be tuples (x, y).
    """
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B
    c = np.array(c)  # Point C

    ba = a - b  # Vector BA
    bc = c - b  # Vector BC

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

# Function to check angles against the correct ranges and provide feedback
def check_pose_angles(angles):
    feedback = []
    is_correct = True

    for joint, angle in angles.items():
        min_angle, max_angle = CORRECT_ANGLES[joint]
        if not (min_angle <= angle <= max_angle):
            feedback.append(f"{joint.replace('_', ' ').capitalize()} is incorrect. Adjust to {min_angle}-{max_angle} degrees.")
            is_correct = False

    return is_correct, feedback

# Main function to process the video feed
def process_video():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Perform pose detection
            results = pose.process(rgb_frame)

            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for angle calculations
                def landmark_coords(index):
                    return (landmarks[index].x, landmarks[index].y)

                left_elbow = calculate_angle(
                    landmark_coords(11), landmark_coords(13), landmark_coords(15)
                )
                right_elbow = calculate_angle(
                    landmark_coords(12), landmark_coords(14), landmark_coords(16)
                )
                left_knee = calculate_angle(
                    landmark_coords(23), landmark_coords(25), landmark_coords(27)
                )
                right_knee = calculate_angle(
                    landmark_coords(24), landmark_coords(26), landmark_coords(28)
                )

                # Create a dictionary of joint angles
                angles = {
                    "left_elbow": left_elbow,
                    "right_elbow": right_elbow,
                    "left_knee": left_knee,
                    "right_knee": right_knee,
                }

                # Check if the pose is correct
                is_correct, feedback = check_pose_angles(angles)

                # Display feedback and angles on the video feed
                for joint, angle in angles.items():
                    cv2.putText(frame, f"{joint.replace('_', ' ').capitalize()}: {int(angle)}°",
                                (10, 30 + list(angles.keys()).index(joint) * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if is_correct:
                    cv2.putText(frame, "Correct Pose!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Incorrect Pose!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3, cv2.LINE_AA)
                    for idx, suggestion in enumerate(feedback):
                        cv2.putText(frame, suggestion, (10, 180 + idx * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow("Yoga Pose Feedback", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    process_video()
