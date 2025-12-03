import pandas as pd

df = pd.read_csv('data/kaggle_dataset/WineQT.csv')

num_rows = len(df[df['quality'] == 4])

print(num_rows)