from flask import Flask, render_template, request, jsonify
from chat import get_response
import os
from flask_cors import CORS
import nltk

# ------------------ NLTK Setup ------------------
# Download 'punkt' inside project if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

# Add the local download folder to NLTK paths
nltk.data.path.append('./nltk_data')

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app)

# ------------------ Routes ------------------
@app.get("/")
def index_get():
    return render_template("base.html")  # Make sure templates/base.html exists

@app.post("/predict")
def predict():
    data = request.get_json()
    text = data.get("message") if data else None

    # Validate input
    if not text or text.strip() == "":
        return jsonify({'answer': "Please enter a valid message."})

    # Get response from chatbot
    response = get_response(text)
    return jsonify({'answer': response})

# ------------------ Run App ------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("DEBUG", "False") == "True"
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT automatically
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
