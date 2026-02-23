import cv2
import joblib
import time
import pandas as pd
import warnings
from src.posture_sensor import PostureSensor
from src.brain import MoodBrain

# 1. SILENCE THE NOISE: This stops the flooding of warnings in your terminal
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Load AI Brain and Models
    print("Loading AI Brain and Models...")
    model = joblib.load('models/stress_model.pkl')
    sensor = PostureSensor()
    brain = MoodBrain()
    cap = cv2.VideoCapture(0)

    # Setup Intervention Logic
    slouch_start_time = None
    trigger_threshold = 10  # Seconds of slouching before AI intervenes
    last_intervention_time = 0
    cooldown_period = 60    # Don't nag more than once a minute

    print("--- MoodSync AI Active: Monitoring Posture ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        score, debug_frame = sensor.get_slouch_score(frame)

        if score:
            # 2. FIX THE FORMAT: Convert score to a DataFrame to match training data
            # This stops the 'feature names' warning
            score_df = pd.DataFrame([[score]], columns=['score'])
            prediction = model.predict(score_df)[0]
            
            if prediction == 1:  # SLOUCHING
                status = "SLOUCHING"
                color = (0, 0, 255) # Red
                
                if slouch_start_time is None:
                    slouch_start_time = time.time()
                
                elapsed = time.time() - slouch_start_time
                
                # Check if it's time for Gemini to speak
                if elapsed > trigger_threshold:
                    current_time = time.time()
                    if (current_time - last_intervention_time) > cooldown_period:
                        print("\n" + "="*40)
                        print("!!! TRIGGERING GEMINI INTERVENTION !!!")
                        
                        # Get the witty response from our brain.py
                        msg = brain.get_intervention("physical fatigue and poor posture")
                        
                        print(f"Gemini says: {msg}")
                        print("="*40 + "\n")
                        
                        last_intervention_time = current_time
            
            else:  # GOOD POSTURE
                status = "Good"
                color = (0, 255, 0) # Green
                slouch_start_time = None

            # Visual Feedback on the webcam window
            cv2.putText(debug_frame, f"State: {status}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if slouch_start_time:
                timer = int(time.time() - slouch_start_time)
                cv2.putText(debug_frame, f"Slouch Timer: {timer}s", (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("MoodSync AI", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()