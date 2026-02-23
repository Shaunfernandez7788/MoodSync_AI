import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class MoodBrain:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_intervention(self, state_data):
        prompt = f"The user is {state_data}. Give a 1-sentence witty reminder to sit up."

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",   # ✅ Correct model for new SDK
                contents=prompt
            )

            return response.text

        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    brain = MoodBrain()
    print(brain.get_intervention("slouching"))