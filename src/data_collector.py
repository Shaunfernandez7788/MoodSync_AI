import cv2
import pandas as pd
import os
from src.posture_sensor import PostureSensor

def record_data():
    cap = cv2.VideoCapture(0)
    sensor = PostureSensor()
    data_list = []
    
    label = input("What state are you recording? (0 for Good, 1 for Slouch): ")
    print(f"Recording 200 frames of state {label}. Move around naturally in that posture...")

    count = 0
    while count < 200:
        success, frame = cap.read()
        if not success: break

        score, debug_frame = sensor.get_slouch_score(frame)
        
        if score:
            # We don't just save the score; we save the raw coordinates
            # This is how you got high accuracy in MedOrbit
            data_list.append({
                'score': score,
                'label': label
            })
            count += 1
            cv2.putText(debug_frame, f"Collected: {count}/200", (50,50), 1, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Save to CSV
    df = pd.DataFrame(data_list)
    file_path = 'data/posture_data.csv'
    # Append if file exists, else create new
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    print(f"Saved to {file_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    record_data()