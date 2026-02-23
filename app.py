import streamlit as st
import cv2
import joblib
import pandas as pd
import time
import warnings
import av
from src.posture_sensor import PostureSensor
from src.brain import MoodBrain
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MoodSync AI Dashboard", layout="wide")
st.title("🧘 MoodSync AI: Smart Posture Coach")

# --- Initialize AI components ---
@st.cache_resource
def load_essentials():
    return MoodBrain(), PostureSensor(), joblib.load('models/stress_model.pkl')

brain, sensor, model = load_essentials()

if 'last_msg' not in st.session_state:
    st.session_state.last_msg = "Sit straight to keep me happy!"

class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.slouch_start = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Simple Posture Analysis
        score, debug_frame = sensor.get_slouch_score(img)
        
        if score:
            score_df = pd.DataFrame([[score]], columns=['score'])
            prediction = model.predict(score_df)[0]
            
            if prediction == 1:
                status, color = "SLOUCHING", (0, 0, 255)
            else:
                status, color = "Good", (0, 255, 0)

            cv2.putText(debug_frame, f"Status: {status}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Use same format as input for maximum compatibility
            return av.VideoFrame.from_ndarray(debug_frame, format="bgr24")
        
        return frame

col1, col2 = st.columns([2, 1])

with col1:
    # UPDATED SECTION: Multi-server STUN/TURN configuration
    webrtc_streamer(
        key="moodsync",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                # Free TURN relay for strict firewalls
                {
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]}
        ),
        video_processor_factory=PostureProcessor,
        async_processing=True,
    )

with col2:
    st.subheader("🤖 AI Intervention")
    st.info(st.session_state.last_msg)
    if st.button("Get Fresh Motivation"):
        with st.spinner("Thinking..."):
            # Ensure the state_data matches the expected input for MoodBrain
            st.session_state.last_msg = brain.get_intervention("slouching")
            st.rerun()