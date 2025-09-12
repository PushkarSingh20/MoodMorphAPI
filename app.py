import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from a .env file in the same directory
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow your frontend
# to make requests to this backend.
CORS(app)

# --- Configuration ---
# Get the Gemini API key from the environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Construct the correct, working URL for the Gemini API endpoint
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# This is the specific set of instructions we will send to the AI model
# to ensure we only get back "happy" or "sad".
SYSTEM_PROMPT = """
You are a highly specialized sentiment analysis model. Your only function is to analyze the following text and classify its dominant emotion as either "happy" or "sad".

Your response MUST be a single word, with no extra formatting, punctuation, or explanation. The only valid responses are the literal strings "happy" or "sad".
"""

@app.route("/")
def home():
    """Provides a simple welcome message to confirm the API is running."""
    return "✅ Python Sentiment Analysis API is running. Use POST /api/sentiment."


@app.route('/api/sentiment', methods=['POST'])
def get_sentiment_route():
    """
    This is the main endpoint that receives user text, sends it to the Gemini API
    for analysis, and returns the resulting sentiment.
    """
    # If the API key is not set, return a mock response for local testing.
    # This prevents unnecessary API calls and avoids rate-limiting issues.
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found. Returning a mock sentiment.")
        # Simple logic for the mock response
        mock_message = request.get_json().get('message', '').lower()
        sentiment = "happy" if "good" in mock_message or "great" in mock_message else "sad"
        return jsonify({'sentiment': sentiment})

    # Get the JSON data from the incoming POST request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required in the request body'}), 400

    user_message = data['message']

    try:
        # This is the data structure (payload) the Gemini API expects.
        # It includes our system prompt to control the output.
        payload = {
            "contents": [{"parts": [{"text": user_message}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}
        }
        
        # Make the POST request to the Gemini API
        response = requests.post(GEMINI_API_URL, json=payload)
        # This will automatically raise an error for bad responses (like 404, 500)
        response.raise_for_status()

        response_data = response.json()
        
        # Carefully extract the single word response from the complex JSON structure
        sentiment = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()

        # Return the clean sentiment to the frontend
        return jsonify({'sentiment': sentiment})

    except requests.exceptions.HTTPError as e:
        # Handle specific errors from the API, like rate limiting (429) or auth issues (403)
        print(f"HTTP Error calling Gemini API: {e.response.text}")
        return jsonify({'error': f'API request failed with status {e.response.status_code}'}), e.response.status_code
    except (KeyError, IndexError):
        # Handle cases where the response from Gemini is not what we expect
        print(f"Error parsing Gemini API response: {response_data}")
        return jsonify({'error': 'Invalid response structure from sentiment service'}), 500
    except requests.exceptions.RequestException as e:
        # Handle network-level errors (e.g., cannot connect to the API)
        print(f"Network error calling Gemini API: {e}")
        return jsonify({'error': 'Failed to connect to the sentiment analysis service'}), 500

# This allows you to run the server directly using "python app.py"
if __name__ == '__main__':
    # Runs the app on localhost, port 5000, with debugging features enabled.
    app.run(port=5000, debug=True)

