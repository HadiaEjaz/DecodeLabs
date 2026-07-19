"""
app.py
The Flask web server. This is what makes the chatbot 'live' on the web:
  - GET  /            -> serves the HTML page (the frontend)
  - POST /api/chat     -> receives a user message, runs it through
                           chatbot_logic.process_message(), returns JSON

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
from chatbot_logic import process_message

app = Flask(__name__)


@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    # The frontend sends JSON like: {"message": "hello"}
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    # This is the actual Python logic running - not a JS copy of it
    result = process_message(user_message)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
