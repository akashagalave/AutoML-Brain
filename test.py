import pandas as pd

train = pd.read_csv("data/interim/train_cleaned.csv")
test = pd.read_csv("data/interim/test_cleaned.csv")

print(train["tenure"].describe())
print(test["tenure"].describe())