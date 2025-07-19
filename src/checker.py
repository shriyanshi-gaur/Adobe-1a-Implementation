import pandas as pd

df = pd.read_csv("../data/processed/training_data.csv")
print(df['label'].value_counts())
