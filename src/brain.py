import os
import streamlit as st
from google import genai
from dotenv import load_dotenv

# Only load .env if it exists (for local testing)
if os.path.exists(".env"):
    load_dotenv()

class MoodBrain:
    def __init__(self):
        # Streamlit Cloud uses st.secrets, but os.getenv also works if configured in 'Secrets'
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in Streamlit Secrets.")
            
        self.client = genai.Client(api_key=api_key)

    def get_intervention(self, state_data):
        prompt = f"The user is {state_data}. Give a 1-sentence witty reminder to sit up."

        try:
            # Using the stable 2.0-flash model ID
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt
            )

            return response.text

        except Exception as e:
            return f"AI is offline, but I still see you slouching! (Error: {e})"

if __name__ == "__main__":
    brain = MoodBrain()
    print(brain.get_intervention("slouching"))