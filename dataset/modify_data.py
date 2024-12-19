import pandas as pd
train = pd.read_csv("data/raw/data.csv")
print(train.head())
print(len(train[train['subreddit']=='sydney']))