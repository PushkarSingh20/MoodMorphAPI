from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# API Key from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://api.gemini-pro.ai/v1/chat/completions"


# Home route (to check if server is running)
@app.route("/")
def home():
    return "✅ Sentiment Analysis API is running. Use POST /sentiment with JSON {\"message\": \"your text\"}."


# Sentiment analysis route
@app.route("/sentiment", methods=["POST"])
def get_sentiment():
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Call Gemini API
        response = requests.post(
            GEMINI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GEMINI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gemini-pro",
                "messages": [
                    {"role": "system", "content": "You are a sentiment classifier. Only answer 'happy' or 'sad'."},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 5
            }
        )

        result = response.json()

        # Extract sentiment safely
        sentiment = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )

        # Ensure only 'happy' or 'sad' gets returned
        if sentiment not in ["happy", "sad"]:
            sentiment = "happy"  # default fallback

        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e), "sentiment": "happy"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

