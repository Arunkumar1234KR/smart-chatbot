from flask import Flask, render_template, request, jsonify
from chat import get_response
import os
from flask_cors import CORS
import nltk

# ------------------ NLTK Setup ------------------
# Create a folder for nltk data
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Add it to NLTK search paths
nltk.data.path.append(nltk_data_dir)

# Download required resources if not already present
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

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
