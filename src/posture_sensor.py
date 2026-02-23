import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

class PostureSensor:
    def __init__(self):
        # Path to the model file you downloaded
        model_path = 'models/pose_landmarker.task'
        
        # Setup the new MediaPipe Tasks options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        # Initialize the detector
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def get_slouch_score(self, frame):
        # MediaPipe requires a specific Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # New API requires a timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Perform the detection
        result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        score = None
        if result.pose_landmarks:
            # Get landmarks for the first person detected
            landmarks = result.pose_landmarks[0]
            
            # --- IMPROVED LOGIC: Multi-Point Averaging ---
            # Landmarks: 7=L_Ear, 8=R_Ear, 11=L_Shoulder, 12=R_Shoulder, 0=Nose
            left_gap = landmarks[11].y - landmarks[7].y
            right_gap = landmarks[12].y - landmarks[8].y
            
            # Average gap for better stability
            score = (left_gap + right_gap) / 2
            
            # Forward Head Posture check: 
            # If your nose (0) is significantly lower than your ears, you are leaning forward
            nose_drop = landmarks[0].y - ((landmarks[7].y + landmarks[8].y) / 2)
            
            # Combine gaps with nose position for the final "Accuracy" score
            # We subtract nose_drop because as you lean forward, your nose goes "down" (higher Y value)
            score = score - (nose_drop * 0.5)

            # Draw visual circles for your feedback
            h, w, _ = frame.shape
            # Draw Shoulders (Blue)
            cv2.circle(frame, (int(landmarks[11].x * w), int(landmarks[11].y * h)), 6, (255, 0, 0), -1)
            cv2.circle(frame, (int(landmarks[12].x * w), int(landmarks[12].y * h)), 6, (255, 0, 0), -1)
            # Draw Ears (Green)
            cv2.circle(frame, (int(landmarks[7].x * w), int(landmarks[7].y * h)), 6, (0, 255, 0), -1)
            cv2.circle(frame, (int(landmarks[8].x * w), int(landmarks[8].y * h)), 6, (0, 255, 0), -1)
            # Draw Nose (Yellow)
            cv2.circle(frame, (int(landmarks[0].x * w), int(landmarks[0].y * h)), 6, (0, 255, 255), -1)

        return score, frame