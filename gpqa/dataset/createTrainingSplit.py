from pathlib import Path
import pandas as pd
import re

# Get GPQA extended and select 20% (round down) for training set.
# Output columns Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Explanation to gpqa/dataset/gpqa_training_split.csv
# The rest of GPQA extended goes to gpqa/dataset/gpqa_eval_split.csv, again with the same columns.

IN_FILE = "gpqa\dataset\gpqa_extended.csv"
TRAIN_OUT = "gpqa\dataset\gpqa_training_split.csv"
EVAL_OUT = "gpqa\dataset\gpqa_eval_split.csv"

DESIRED_COLS = ["Question","Correct Answer","Incorrect Answer 1","Incorrect Answer 2","Incorrect Answer 3","Explanation"]
def main():
    # Read the input CSV
    df = pd.read_csv(IN_FILE)

    # Select only the desired columns
    df.columns = df.columns.str.strip()
    
    # now safe to select by exact names in DESIRED_COLS
    df = df[DESIRED_COLS]
    print(df.columns.tolist())
    print(df.head())

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split index (20% for training, round down)
    train_size = int(len(df) * 0.2)

    # Split the dataframe
    train_df = df.iloc[:train_size]
    eval_df = df.iloc[train_size:]

    # Write to output CSVs
    train_df.to_csv(TRAIN_OUT, index=False)
    eval_df.to_csv(EVAL_OUT, index=False)

if __name__ == "__main__":
    main()
