
import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. LOAD COMBINED DATASET

df = pd.read_csv("combined_dataset.csv")
print("Total samples:", len(df))



# 2. CLEANING FUNCTION (same used in Flask!)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)



# 3. BALANCE CLASSES STRICTLY

fake_df = df[df["label"] == 0]
real_df = df[df["label"] == 1]

min_size = min(len(fake_df), len(real_df))

fake_df = fake_df.sample(min_size, random_state=42)
real_df = real_df.sample(min_size, random_state=42)

df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42)
print("Balanced dataset size:", len(df))



# 4. FIX LENGTH BIAS (critical!)

df["length"] = df["text"].str.split().apply(len)

# Cap maximum article length to reduce overfitting
df["text"] = df["text"].apply(lambda t: " ".join(t.split()[:250]))



# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)



# 6. TF-IDF VECTORIZER (BEST CONFIG)

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,        # removes rare noise
    max_df=0.90      # removes extremely common words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# 7. LOGISTIC REGRESSION MODEL (BEST for TF-IDF)

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",    # prevents bias toward one label
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)


# 8. EVALUATION

preds = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))



# 9. SAVE MODEL + VECTORIZER

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")   # keep filename same for Flask
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully!")



