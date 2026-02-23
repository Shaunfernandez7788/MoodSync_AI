import streamlit as st
import cv2
import joblib
import pandas as pd
import time
import warnings
from src.posture_sensor import PostureSensor
from src.brain import MoodBrain

# Ignore unpickle warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="MoodSync AI Dashboard", layout="wide")

st.title("🧘 MoodSync AI: Smart Posture Coach")

# --- Initialize AI components in Session State (to keep them alive) ---
if 'brain' not in st.session_state:
    st.session_state.brain = MoodBrain()
if 'slouch_start' not in st.session_state:
    st.session_state.slouch_start = None
if 'last_msg' not in st.session_state:
    st.session_state.last_msg = "Sit straight to keep me happy!"

# Load Models
model = joblib.load('models/stress_model.pkl')
sensor = PostureSensor()

col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Start AI Monitoring', value=True)
    FRAME_WINDOW = st.image([]) 

with col2:
    st.subheader("🤖 AI Intervention")
    # This is where the Gemini message will appear
    msg_container = st.empty()
    msg_container.info(st.session_state.last_msg)

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        score, debug_frame = sensor.get_slouch_score(frame)
        
        if score:
            score_df = pd.DataFrame([[score]], columns=['score'])
            prediction = model.predict(score_df)[0]
            
            if prediction == 1: # SLOUCHING
                if st.session_state.slouch_start is None:
                    st.session_state.slouch_start = time.time()
                
                elapsed = time.time() - st.session_state.slouch_start
                status = f"SLOUCHING ({int(elapsed)}s)"
                color = (0, 0, 255)
                
                # TRIGGER GEMINI at 10 seconds
                if elapsed > 10:
                    with st.spinner("Gemini is thinking..."):
                        new_msg = st.session_state.brain.get_intervention("poor posture")
                        st.session_state.last_msg = new_msg
                        msg_container.warning(f"**Gemini:** {new_msg}")
                        st.session_state.slouch_start = time.time() - 100 # Cooldown
            else:
                status = "Good"
                color = (0, 255, 0)
                st.session_state.slouch_start = None
                msg_container.info(st.session_state.last_msg)

            cv2.putText(debug_frame, f"Status: {status}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert and Update Dashboard
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(debug_frame)
else:
    st.write("Camera Stopped.")