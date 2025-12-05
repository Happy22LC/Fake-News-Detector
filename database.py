import sqlite3
from datetime import datetime

DB_NAME = "database.db"

# ------------------------------
# Create database + table
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_input TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ------------------------------
# Save prediction into database
# ------------------------------
def save_prediction(text, label, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions (text_input, predicted_label, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (text, label, confidence, timestamp))

    conn.commit()
    conn.close()


# ------------------------------
# Fetch all saved predictions
# ------------------------------
def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows
