import pandas as pd
import numpy as np
import random
import argparse

def process_data(df_path, save_path, sub_sample = None, test_sample_size = None, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)

    df = pd.read_csv(df_path)

    if sub_sample:
        sub_df = df.sample(n=sub_sample, replace=False, random_state=random_state)
        curated_df = df.drop(sub_df.index)
        sub_df.reset_index(inplace=True, drop=True)
        sub_test = curated_df.sample(n=test_sample_size, random_state=random_state)
        sub_test.reset_index(inplace=True, drop=True)
        sub_df.to_csv(save_path+f"df{sub_sample}.csv", index=False)
        sub_test.to_csv(save_path+f"df{sub_sample}_test.csv", index=False)
    else:
        train_size = int(len(df) * 0.8)
        test_size = int(len(df) * 0.2)
        sub_df = df.sample(n=train_size, replace=False, random_state=random_state)
        curated_df = df.drop(sub_df.index)
        sub_df.reset_index(inplace=True, drop=True)
        sub_test = curated_df.sample(n=test_size, random_state=random_state)
        sub_test.reset_index(inplace=True, drop=True)
        sub_df.to_csv(save_path+f"df{sub_sample}.csv", index=False)
        sub_test.to_csv(save_path+f"df{sub_sample}_test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data with given parameters: ")
    parser.add_argument("--datapath", type=str, required=True, help="Path of the original dataset")
    parser.add_argument("--savepath", type=str, required=True, help="Save the processed dataset")
    parser.add_argument("--subsample", type=int, required=False, help = "Sample size of training dataset")
    parser.add_argument("--test_sample_size", type=int, required=False, help="Sample size of test data")
    parser.add_argument("--random_state", type=int, required=False, help="Random seed")

    args = parser.parse_args()
    process_data(args.datapath, args.savepath, args.subsample, args.test_sample_size)





