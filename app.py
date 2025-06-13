from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)  # Enables requests from Flutter Web

# NVIDIA Nemotron client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# ✅ 1. Simple AI Suggestion for AQI/temp/humidity
@app.route("/ai-suggestion", methods=["POST"])
def ai_suggestion():
    data = request.get_json()
    aqi = data.get("aqi", 0)
    temp = data.get("temp", 0)
    humidity = data.get("humidity", 0)

    messages = [
        {
            "role": "system",
            "content": "You're an expert in air quality, health, and environment safety. Reply with a short useful tip."
        },
        {
            "role": "user",
            "content": f"AQI: {aqi}, Temperature: {temp}°C, Humidity: {humidity}%. What should I do?"
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=300
        )
        suggestion = completion.choices[0].message.content
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ 2. AI Chatbot: Ask any question
@app.route("/ai-chat", methods=["POST"])
def ai_chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    messages = [
        {"role": "system", "content": "You're a helpful AI assistant specializing in environment, pollution control, and health."},
        {"role": "user", "content": user_input}
    ]

    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=600
        )
        reply = completion.choices[0].message.content
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Default route
@app.route("/")
def home():
    return "Soot Sweeper™ AI Backend is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
