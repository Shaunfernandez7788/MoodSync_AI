import os
import streamlit as st
from google import genai


class MoodBrain:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in Streamlit Secrets.")

        self.client = genai.Client(api_key=api_key)

    def get_intervention(self, state_data):
        prompt = f"The user is {state_data}. Give a 1-sentence witty reminder to sit up."

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text

        except Exception:
            return "AI is offline, but I still see you slouching!"