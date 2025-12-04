import os
import pandas as pd

# 1. PATH  DATASET FOLDER

BASE_DIR = "datasets"   # Your folder name


# 2. HELPER: READ A SINGLE .txt FILE

def read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except:
        return ""

# 3. SCAN FOLDERS FOR FAKE & REAL NEWS

def load_dataset():
    data = []

    for root, dirs, files in os.walk(BASE_DIR):
        print("ROOT →", repr(root))  # <— ADD THIS
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)

                # LABEL LOGIC -----------------------------
                folder = os.path.basename(root).lower()

                if folder in ["fake", "fake_articles", "Catalog - Fake Articles"]:
                    label = 0

                elif folder in ["real", "legit", "real_articles", "Catalog - Real Articles"]:
                    label = 1

                else:
                    continue

                text = read_text_file(filepath)

                if len(text) > 20:   # avoid empty files
                    data.append([text, label])
                    print("Loaded:", filepath, "| Folder:", folder, "| Label:", label)


    return data


# 4. LOAD AND SAVE

all_data = load_dataset()

print("Loaded samples:", len(all_data))

df = pd.DataFrame(all_data, columns=["text", "label"])

df.to_csv("combined_dataset.csv", index=False, encoding="utf-8")

print("\ncombined_dataset.csv CREATED SUCCESSFULLY!")
