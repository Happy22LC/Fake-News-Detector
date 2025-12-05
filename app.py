from flask import Flask, render_template, request, jsonify
import joblib
import os
import re
from database import init_db, save_prediction, get_all_predictions
app = Flask(__name__)

# Initialize database on startup
init_db()

# Load model + vectorizer
model = joblib.load("models/xgboost_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# Cleaning function

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    return text.lower().strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text")

    if not text or text.strip() == "":
        return jsonify({"error": "Text cannot be empty"}), 400

    # CLEAN the input (IMPORTANT!)
    print("\nRAW TEXT:", text)
    cleaned = clean_text(text)
    print("CLEANED TEXT:", cleaned)

    # Transform with the TF-IDF vectorizer
    tfidf = vectorizer.transform([cleaned])

    # Predict
    pred = int(model.predict(tfidf)[0])

    # Get probability
    proba = model.predict_proba(tfidf)[0]
    prob = float(proba[pred])   #this is probability value

    label = "FAKE NEWS" if pred == 0 else "REAL NEWS"


    # Save result into SQLite
    save_prediction(text, label, prob * 100)


    return jsonify({
        "prediction": label,
        "confidence": round(prob * 100, 2)
    })


@app.route("/logs")
def logs_page():
    logs = get_all_predictions()
    return render_template("logs.html", logs=logs)

if __name__ == "__main__":
    app.run(debug=True)





"""""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import sqlite3
import datetime
import os

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
model = joblib.load("models/xgboost_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Database init
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("""
        #CREATE TABLE IF NOT EXISTS logs (
            #id INTEGER PRIMARY KEY AUTOINCREMENT,
            #text TEXT,
            #prediction TEXT,
            #confidence REAL,
            #timestamp TEXT
        #)
""")
    conn.commit()
    conn.close()

init_db()

print("MODEL LOADED SUCCESSFULLY")
print("Model file timestamp:", os.path.getmtime("models/xgboost_model.pkl"))
print(">>> Reached ROUTE section — Flask is loading routes now")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("news_text")

    if not text or text.strip() == "":
        return jsonify({"error": "Text cannot be empty"}), 400

    tfidf = vectorizer.transform([text])

    prediction = model.predict(tfidf)[0]
    confidence = max(model.predict_proba(tfidf)[0])

    # Convert to Python native types
    prediction = int(prediction)
    #prediction = "FAKE NEWS" if int(prediction) == 0 else "REAL NEWS"
    confidence = float(round(confidence * 100, 2))

    # Log to DB
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (text, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (text, prediction, confidence, str(datetime.datetime.now()))
    )
    conn.commit()
    conn.close()

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
"""""