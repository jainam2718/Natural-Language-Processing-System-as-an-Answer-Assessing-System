import pandas as pd

df = pd.read_parquet("train-00000-of-00001.parquet")
df.to_csv("biosses_dataset.csv")