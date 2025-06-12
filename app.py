from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")  # Keep secret
)

@app.route("/ai-suggestion", methods=["POST"])
def suggest():
    data = request.get_json()
    aqi = data.get("aqi")
    temp = data.get("temp")
    humidity = data.get("humidity")

    messages = [
        {
            "role": "system",
            "content": "You're an expert in pollution control. Give 1-line smart tips for health and environment."
        },
        {
            "role": "user",
            "content": f"AQI: {aqi}, Temp: {temp}°C, Humidity: {humidity}% – what should user do now?"
        }
    ]

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=300,
    )

    suggestion = response.choices[0].message.content
    return jsonify({"suggestion": suggestion})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
