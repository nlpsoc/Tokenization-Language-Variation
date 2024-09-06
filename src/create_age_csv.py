from styletokenizer.utility.env_variables import set_cache

set_cache()

import pandas as pd
import os
from datasets import load_dataset
import csv

# Load the Blog Authorship Corpus dataset from Hugging Face
dataset = load_dataset('barilan/blog_authorship_corpus')


# Function to add label based on age
def assign_label(age):
    if 13 <= age <= 17:
        return '10s'
    elif 23 <= age <= 27:
        return '20s'
    elif 33 <= age <= 47:
        return '30s'
    else:
        return None


# Output directory
output_dir = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create a DataFrame from each split, add the label column, and save to TSV
df_splits = {}
for split in dataset:
    df = pd.DataFrame(dataset[split])

    # Add the 'label' column
    df['label'] = df['age'].apply(assign_label)

    # Check for invalid ages and remove them
    valid_ages = df['label'].notnull()
    # check if any invalid ages
    if len(valid_ages) != len(df):
        print(f"Invalid ages in {split}: {df[~valid_ages]['age'].unique()}")
    df = df[valid_ages]

    # check for invalid texts
    valid_texts = df['text'].notnull()
    if len(valid_texts) != len(df):
        print(f"Invalid texts in {split}: {df[~valid_texts]['text'].unique()}")

    # Save the DataFrame to a TSV file
    output_path = os.path.join(output_dir, f"{split}.csv")
    df.to_csv(output_path, index=False, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {split} to {output_path}")

    # Verify unique ages
    unique_ages = df['age'].unique()
    print(f"Unique ages in {split}: {unique_ages}")