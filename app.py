from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Read NVIDIA API key from environment
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# ✅ Create OpenAI client using NVIDIA base
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key
)

# ✅ Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route("/")
def home():
    return "🟢 Soot Sweeper AI Backend is running."

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        print("🧠 AI Prompt:", prompt)  # Debug log

        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=1024
        )

        reply = completion.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        print("🔥 AI Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
