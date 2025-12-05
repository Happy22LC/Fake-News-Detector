from flask import Flask, render_template, request, jsonify
import joblib
import re
from database import init_db, save_prediction, get_all_predictions
app = Flask(__name__)

# initialize database
init_db()

# Load model and vectorizer
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

    # Clean the input text
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
