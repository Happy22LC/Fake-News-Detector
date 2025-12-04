import pandas as pd

# This is a debug file to check dataset
df = pd.read_csv("combined_dataset.csv")
print(df['label'].value_counts())
