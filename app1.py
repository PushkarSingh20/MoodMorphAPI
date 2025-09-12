import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow frontend requests
CORS(app)

# --- Configuration based on debugging session ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using the final, correct endpoint and model from the debug session
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"


@app.route("/")
def home():
    """Provides a simple status check for the API."""
    return "✅ Sentiment and Chat API is running. Use POST /api/sentiment and POST /api/chat."


@app.route('/api/sentiment', methods=['POST'])
def get_sentiment_route():
    """Analyzes the sentiment of a message using the Gemini API."""
    
    if not GEMINI_API_KEY:
        print("Warning: API key not found. Returning a mock sentiment.")
        sentiment = "happy" if "good" in request.get_json().get('message', '').lower() else "sad"
        return jsonify({'sentiment': sentiment})

    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    # System prompt to ensure a clean, single-word response
    system_prompt = 'You are a highly specialized sentiment analysis model. Your only function is to analyze the following text and classify its dominant emotion as either "happy" or "sad". Your response MUST be a single word, with no extra formatting, punctuation, or explanation. The only valid responses are the literal strings "happy" or "sad".'

    try:
        payload = {
            "contents": [{"parts": [{"text": message}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()

        response_data = response.json()
        sentiment = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()

        return jsonify({'sentiment': sentiment})

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling Gemini API: {e.response.text}")
        return jsonify({'error': f'API request failed with status {e.response.status_code}'}), e.response.status_code
    except (KeyError, IndexError):
        print(f"Error parsing Gemini API response: {response.json()}")
        return jsonify({'error': 'Invalid response structure from sentiment service'}), 500
    except requests.exceptions.RequestException as e:
        print(f"Network error calling Gemini API: {e}")
        return jsonify({'error': 'Failed to connect to the sentiment analysis service'}), 500


# --- NEW CHAT ENDPOINT ---
@app.route('/api/chat', methods=['POST'])
def continue_chat_route():
    """Handles general conversational messages with the AI."""
    
    if not GEMINI_API_KEY:
        return jsonify({'response': "I'm in offline mode right now, but we can chat more later!"})

    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    # A simple system prompt for the chat AI
    system_prompt = "You are a friendly and supportive AI assistant. Keep your responses concise and encouraging."

    try:
        payload = {
            "contents": [{"parts": [{"text": message}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]}
        }
        
        response = requests.post(GEMINI_API_URL, json=payload)
        response.raise_for_status()

        response_data = response.json()
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text'].strip()

        return jsonify({'response': ai_response})

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error in chat: {e.response.text}")
        return jsonify({'error': 'The chat service is currently unavailable.'}), e.response.status_code
    except Exception as e:
        print(f"An unexpected error occurred in chat: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
