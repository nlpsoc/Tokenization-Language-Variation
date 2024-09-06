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

    # Save the DataFrame to a TSV file
    output_path = os.path.join(output_dir, f"{split}.csv")
    df.to_csv(output_path, index=False, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {split} to {output_path}")

    # try loading the saved file
    df = pd.read_csv(output_path)
    nan_values = df['text'].isna()
    print(f"Nan values at: {df[nan_values].index}")
    print(f"Overwriting save without nan values to {output_path}")
    # remove rows with nan values
    df = df[~nan_values]
    df.to_csv(output_path, index=False, quotechar='"', quoting=csv.QUOTE_MINIMAL)
