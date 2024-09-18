import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import threading
import time
import requests
from collections import deque
import logging

app = Flask(__name__)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_sessions = {}  # Dictionary to store chat sessions per user
SESSION_TIMEOUT = 3600  # 1 hour timeout for sessions

def cleanup_sessions():
    """Remove expired sessions."""
    current_time = time.time()
    for user_id in list(chat_sessions.keys()):
        if current_time - chat_sessions[user_id]['last_activity'] > SESSION_TIMEOUT:
            del chat_sessions[user_id]

# Predefined chat history for Bengali to Banglish translation
predefined_history = [
    {
        "role": "user",
        "parts": [
            "আমি যে শব্দগুলো দিব ওইগুলোকে বাংলা থেকে বাংলিশ এর পরিণত কর। যেমন আমার সোনার বাংলাকে Amer Sonar Bangla রকম ভাবে রূপান্তর করে দিবে। আর এখানে উত্তর হিসেবে শুধু বাংলিশ টি দিবে আর কোন কিছু দিবে না আমি যাই লিখব তাকেই বাংলিশ করে দিবে অতিরিক্ত কোন শব্দ লিখবে না শুধু বাংলিশটুকু দিবে।",
        ],
    },
    {
        "role": "model",
        "parts": [
            "আপনি যে শব্দগুলো দিবেন আমি সেগুলো বাংলিশ করে দিবো। \n",
        ],
    },
    {
        "role": "user",
        "parts": [
            "এটি অনেক দুর্ভাগ্যবশত হয়ে গেছে কিছু করার নেই",
        ],
    },
    {
        "role": "model",
        "parts": [
            "Eti onek durvagyoboshot hoye geche kichu karar nei \n",
        ],
    },
    {
        "role": "user",
        "parts": [
            "সে আরো এটি অনেক কিছু করার ছিল যাই হোক কিছু করার নেই এখন যা হওয়ার হয়ে গেছে এখন তুমি বলো কেমন আছো?",
        ],
    },
    {
        "role": "model",
        "parts": [
            "Se aro eti onek kichu karar chilo jai hok kichu karar nei ekon ja howar hoye geche ekon tumi bolo kemon acho? \n",
        ],
    },
]

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q')
    user_id = request.args.get('id')

    if not query or not user_id:
        return jsonify({"error": "Please provide both query and id parameters."}), 400

    try:
        if user_id not in chat_sessions:
            chat_sessions[user_id] = {
                "chat": model.start_chat(history=predefined_history),  # Initialize with predefined history
                "history": deque(maxlen=5),  # Stores the last 5 messages
                "last_activity": time.time()
            }

        chat_session = chat_sessions[user_id]["chat"]
        history = chat_sessions[user_id]["history"]

        # Add the user query to history
        history.append(f"User: {query}")
        response = chat_session.send_message(query)
        # Add the bot response to history
        history.append(f"Bot: {response.text}")

        chat_sessions[user_id]["last_activity"] = time.time()  # Update session activity

        return jsonify({"response": response.text})
    
    except Exception as e:
        logging.error(f"Error during chat processing: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "alive"})

def keep_alive():
    url = "https://gemini-api-5dl7.onrender.com"  # Replace with your actual URL
    while True:
        time.sleep(600)  # Ping every 10 minutes
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Ping successful")
            else:
                print("Ping failed with status code", response.status_code)
        except requests.exceptions.RequestException as e:
            print("Ping failed with exception", e)

if __name__ == '__main__':
    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()
    # Cleanup old sessions every 15 minutes
    threading.Thread(target=lambda: time.sleep(900) or cleanup_sessions(), daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
