"""
Prepare a trimmed IMDB dataset for the sentiment classifier.
"""

import pandas as pd
import json
import os

# Change to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(script_dir))

# Load and sample data
df = pd.read_csv("data/IMDB Dataset.csv")
positive = df[df['sentiment'] == 'positive'].sample(100, random_state=42)
negative = df[df['sentiment'] == 'negative'].sample(100, random_state=42)
balanced = pd.concat([positive, negative]).sample(frac=1, random_state=42)

# Convert to JSON format
data = [
    {"sentence": row['review'].replace('<br />', ' '), 
     "gold_label": "Positive" if row['sentiment'] == 'positive' else "Negative"}
    for _, row in balanced.iterrows()
]

# Save
with open("data/imdb_reviews_sentiment.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Created dataset with {len(data)} reviews")
