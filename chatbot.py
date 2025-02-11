import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

# Load training data from CSV
df = pd.read_csv("chatbot_data.csv")

# Prepare training data
X_train = []
y_train = []
responses = {}

for index, row in df.iterrows():
    patterns = row["patterns"].split("|")  # Split patterns using '|'
    for pattern in patterns:
        X_train.append(pattern)
        y_train.append(row["intent"])
    responses[row["intent"]] = row["response"]

# Convert text to numerical data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train ML model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Function to predict response
def chatbot_response(user_input):
    input_vectorized = vectorizer.transform([user_input])
    intent = model.predict(input_vectorized)[0]
    return responses.get(intent, "Sorry, I don't understand.")

# Flask route to handle chat requests
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
