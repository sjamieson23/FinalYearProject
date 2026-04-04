import pandas as pd
from pathlib import Path

def combine_datasets():
    count = 0
    spearbot_df = pd.DataFrame()
    for path in ["Data/RefinedSpearPhishing/Employees", "Data/RefinedSpearPhishing/Students"]:
        for file in Path(path).glob("*.csv"):
            count += 1
            df = pd.read_csv(file)
            df.columns = df.columns.str.lower()
            if "Safe" in file.name:
                print("good")
                df["label"] = 0
                spearbot_df = pd.concat([spearbot_df, df])
            elif "Phishing" in file.name:
                print("bad")
                df["label"] = 1
                spearbot_df = pd.concat([spearbot_df, df])
            else:
                print(f"Unknown label for file: {file}")
    print(count)
    spearbot_df.to_csv("Data/RefinedSpearPhishing/refined_spearbot_data.csv", index=False)



combine_datasets()
