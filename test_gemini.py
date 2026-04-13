import google.generativeai as genai
import sys
import traceback

genai.configure(api_key='AIzaSyDU9xnhcHo7KfUd5Qs_ocCAH0YoRhQ3D3I')
model = genai.GenerativeModel("gemini-2.5-flash")

try:
    response = model.generate_content('Reply with exactly the word SUCCESS')
    print("API SUCCESS:", response.text)
except Exception as e:
    with open("error.log", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    print("API FAILED! Error written to error.log")
