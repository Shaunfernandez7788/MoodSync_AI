import streamlit as st
import cv2
import joblib
import pandas as pd
import warnings
import av

from src.posture_sensor import PostureSensor
from src.brain import MoodBrain
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

warnings.filterwarnings("ignore")

st.set_page_config(page_title="MoodSync AI Dashboard", layout="wide")
st.title("🧘 MoodSync AI: Smart Posture Coach")

# -----------------------------
# Load AI components (cached)
# -----------------------------
@st.cache_resource
def load_essentials():
    brain = MoodBrain()
    sensor = PostureSensor()
    model = joblib.load("models/stress_model.pkl")
    return brain, sensor, model

brain, sensor, model = load_essentials()

if "last_msg" not in st.session_state:
    st.session_state.last_msg = "Sit straight to keep me happy!"

# -----------------------------
# Video Processor
# -----------------------------
class PostureProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        try:
            score, debug_frame = sensor.get_slouch_score(img)

            if score is not None:
                score_df = pd.DataFrame([[score]], columns=["score"])
                prediction = model.predict(score_df)[0]

                if prediction == 1:
                    status, color = "SLOUCHING", (0, 0, 255)
                else:
                    status, color = "Good", (0, 255, 0)

                cv2.putText(
                    debug_frame,
                    f"Status: {status}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

                return av.VideoFrame.from_ndarray(debug_frame, format="bgr24")

        except Exception as e:
            print("Frame processing error:", e)

        return frame


# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="moodsync",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        video_processor_factory=PostureProcessor,
        async_processing=False,   # IMPORTANT for Cloud stability
    )

with col2:
    st.subheader("🤖 AI Intervention")
    st.info(st.session_state.last_msg)

    if st.button("Get Fresh Motivation"):
        with st.spinner("Thinking..."):
            try:
                st.session_state.last_msg = brain.get_intervention("slouching")
            except Exception:
                st.session_state.last_msg = "Sit straight, superstar!"
            st.rerun()