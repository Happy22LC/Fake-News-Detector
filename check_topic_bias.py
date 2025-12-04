import pandas as pd

#Debug file for check biased topic
df = pd.read_csv("combined_dataset.csv")

check_words = "federal|inflation|employment|reserve"

result = df['text'].str.contains(check_words, case=False).groupby(df['label']).sum()

print(result)
