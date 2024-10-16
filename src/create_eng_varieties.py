"""
    script generated with GPT4 o1-preview on 2024-10-16
"""
import os
import pandas as pd
import gzip
import glob
from sklearn.model_selection import train_test_split

# Define the path to the folder containing the gzipped CSV files
folder_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Varieties/CGLUv5.2/"

# Define the list of possible origins
origins = ['Nigeria', 'Canada', 'United_States', 'Hong_Kong', 'India', 'Pakistan', 'Malaysia',
           'Philippines', 'Singapore', 'Ireland', 'United_Kingdom', 'Australia', 'New_Zealand']


# Function to extract origin from filename
def extract_origin_from_filename(filename):
    for origin in origins:
        origin_formatted = origin.replace('_', ' ')
        if origin_formatted in filename or origin in filename:
            return origin
    return 'Unknown'


# Get the list of all gzipped CSV files in the folder
file_list = glob.glob(os.path.join(folder_path, '*.gz'))

# Dictionary to store total words in each file
file_word_counts = {}

# First, compute the total N_Words in each file
for file in file_list:
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)
        total_words = df['N_Words'].sum()
        file_word_counts[file] = total_words

# Find the minimum total N_Words across all files
min_total_words = min(file_word_counts.values())

# List to store sampled DataFrames
sampled_dfs = []

# For each file, sample rows until cumulative N_Words >= min_total_words
for file in file_list:
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)

        # Shuffle the DataFrame for random sampling
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Initialize cumulative N_Words
        cumulative_words = 0
        sampled_rows = []

        for index, row in df.iterrows():
            cumulative_words += row['N_Words']
            sampled_rows.append(row)
            if cumulative_words >= min_total_words:
                break

        # Create DataFrame from sampled rows
        sampled_df = pd.DataFrame(sampled_rows)

        # Extract origin from filename
        filename = os.path.basename(file)
        origin = extract_origin_from_filename(filename)
        sampled_df['origin'] = origin

        sampled_dfs.append(sampled_df)

# Concatenate all sampled DataFrames
final_df = pd.concat(sampled_dfs, ignore_index=True)

# Split the data into training, development, and test sets (80-10-10 split)
# First, split off the test set from the rest of the data
train_dev_df, test_df = train_test_split(
    final_df,
    test_size=0.10,
    random_state=42,
    stratify=final_df['origin']
)

# Then, split the remaining data into training and development sets
train_df, dev_df = train_test_split(
    train_dev_df,
    test_size=0.1111,  # 0.1111 * 90% ≈ 10% of the total data
    random_state=42,
    stratify=train_dev_df['origin']
)

# Shuffle the datasets to mix origins
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the datasets to CSV files
train_df.to_csv('train.csv', index=False)
dev_df.to_csv('dev.csv', index=False)
test_df.to_csv('test.csv', index=False)