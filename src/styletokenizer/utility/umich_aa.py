import ast
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

DEV_PATH = "../../data/UMich-AA/amazon/eval.jsonl"
COMBINED_PATH = "../../data/UMich-AA/long-reddit/combined.jsonl"


def load_dev_data():
    with open(DEV_PATH, 'r') as f:
        data = [json.loads(line) for line in f]

    # create pandas dataframe
    df = pd.DataFrame(data)
    df['authorIDs'] = df['authorIDs'].apply(lambda x: ast.literal_eval(x)[0])

    return df


# Function to split data into train, dev, and test sets
def stratified_split(df, group_col, train_size=0.6, dev_size=0.2, test_size=0.2, random_state=42):
    # First, ensure each group is represented in each split
    unique_groups = df[group_col].unique()

    # Create placeholders for the splits
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # For each unique group, split their data into train, dev, and test
    for group in unique_groups:
        group_data = df[df[group_col] == group]

        # Split into train and temp (dev + test)
        train_data, temp_data = train_test_split(group_data, test_size=(1 - train_size), random_state=random_state)

        # Split temp into dev and test
        dev_data, test_data = train_test_split(temp_data, test_size=(test_size / (dev_size + test_size)),
                                               random_state=random_state)

        # Append the splits to the respective DataFrames
        train_df = pd.concat([train_df, train_data])
        dev_df = pd.concat([dev_df, dev_data])
        test_df = pd.concat([test_df, test_data])

    return train_df, dev_df, test_df


def split_df(df):
    train, dev, test = stratified_split(df, group_col='authorIDs')
    return train, dev, test
