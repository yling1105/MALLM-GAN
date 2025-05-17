import pandas as pd

df = pd.read_csv("df50.csv")
df_test = pd.read_csv("df50_test.csv")

cols = [
        "age",
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "Income"
    ]
df = df[cols]
df_test = df_test[cols]
df.to_csv("df50.csv", index=False)
df_test.to_csv("df50_test.csv", index=False)