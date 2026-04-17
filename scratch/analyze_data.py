import pandas as pd
import os

files = ['data/train.csv', 'data/test.csv', 'data/sft_train.csv']
for f in files:
    path = os.path.join('d:/Personal/Documents/GitHub/SLM-RL-Comparation', f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"--- {f} ---")
        print(f"Total Rows: {len(df)}")
        if 'nums' in df.columns:
            # Estimate N from nums column
            df['n'] = df['nums'].apply(lambda x: len(str(x).split(',')))
            print("Distribution of N:")
            print(df['n'].value_counts().sort_index().to_dict())
        print("\n")
    else:
        print(f"File not found: {path}")
