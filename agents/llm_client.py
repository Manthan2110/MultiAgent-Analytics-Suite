import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_llm_response(system_prompt, user_prompt):
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite")

    response = model.generate_content([
        {"role": "user", "parts": [system_prompt]},
        {"role": "user", "parts": [user_prompt]},
    ])

    return response.text
