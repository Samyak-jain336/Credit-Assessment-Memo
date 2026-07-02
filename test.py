import google.genai as genai
import os

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

for model in client.models.list():
    if "embed" in model.name.lower():
        print(model.name)