import streamlit as st
import cv2
import joblib
import pandas as pd
import time
import warnings
import numpy as np
from src.posture_sensor import PostureSensor
from src.brain import MoodBrain
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Ignore unpickle warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="MoodSync AI Dashboard", layout="wide")
st.title("🧘 MoodSync AI: Smart Posture Coach")

# --- Initialize AI components in Session State ---
if 'brain' not in st.session_state:
    st.session_state.brain = MoodBrain()
if 'slouch_start' not in st.session_state:
    st.session_state.slouch_start = None
if 'last_msg' not in st.session_state:
    st.session_state.last_msg = "Sit straight to keep me happy!"

# Load Models
model = joblib.load('models/stress_model.pkl')
sensor = PostureSensor()

class PostureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Posture Logic
        score, debug_frame = sensor.get_slouch_score(img)
        
        if score:
            score_df = pd.DataFrame([[score]], columns=['score'])
            prediction = model.predict(score_df)[0]
            
            if prediction == 1: # SLOUCHING
                if st.session_state.slouch_start is None:
                    st.session_state.slouch_start = time.time()
                
                elapsed = time.time() - st.session_state.slouch_start
                status = f"SLOUCHING ({int(elapsed)}s)"
                color = (0, 0, 255)
                
                # Logic to trigger Gemini intervention (updates happen in the UI thread)
                if elapsed > 10:
                    st.session_state.trigger_ai = True
            else:
                status = "Good"
                color = (0, 255, 0)
                st.session_state.slouch_start = None

            cv2.putText(debug_frame, f"Status: {status}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return debug_frame

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="posture-filter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=PostureTransformer,
        async_transform=True,
    )

with col2:
    st.subheader("🤖 AI Intervention")
    msg_container = st.empty()
    
    # Check if we need to trigger Gemini
    if st.session_state.get("trigger_ai"):
        with st.spinner("Gemini is thinking..."):
            new_msg = st.session_state.brain.get_intervention("poor posture")
            st.session_state.last_msg = new_msg
            st.session_state.trigger_ai = False
            st.session_state.slouch_start = time.time() - 100 # Cooldown
            
    msg_container.warning(f"**Gemini:** {st.session_state.last_msg}")