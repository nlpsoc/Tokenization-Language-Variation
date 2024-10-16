"""
    script generated with GPT4 o1-preview on 2024-10-16
"""
import os
import pandas as pd
import gzip
import glob
from sklearn.model_selection import train_test_split

# Define the path to the folder containing the gzipped CSV files
folder_path = 'path_to_your_folder'

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


print("Starting the sampling process...")

# Get the list of all gzipped CSV files in the folder
file_list = glob.glob(os.path.join(folder_path, '*.gz'))

# Dictionary to store total words in each file
file_word_counts = {}

print(f"Found {len(file_list)} files to process.")

# First, compute the total N_Words in each file
print("Calculating total number of words in each file...")
for file in file_list:
    filename = os.path.basename(file)
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)
        total_words = df['N_Words'].sum()
        file_word_counts[file] = total_words
        print(f"File '{filename}': {total_words} words.")

# Find the minimum total N_Words across all files
min_total_words = min(file_word_counts.values())
print(f"Minimum total words across all files: {min_total_words}")

# List to store sampled DataFrames
sampled_dfs = []

# For each file, sample rows until cumulative N_Words >= min_total_words
print("Sampling rows from each file to reach equal total number of words...")
for file in file_list:
    filename = os.path.basename(file)
    print(f"Processing file '{filename}'...")
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
        print(f"Sampled {len(sampled_df)} rows from '{filename}'.")

        # Extract origin from filename
        origin = extract_origin_from_filename(filename)
        sampled_df['origin'] = origin

        sampled_dfs.append(sampled_df)

print("Concatenating all sampled data into a single DataFrame...")
# Concatenate all sampled DataFrames
final_df = pd.concat(sampled_dfs, ignore_index=True)
print(f"Total number of samples after concatenation: {len(final_df)}")

print("Splitting the data into training, development, and test sets (80-10-10 split)...")
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
print("Shuffling the training, development, and test sets...")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Saving the datasets to CSV files...")
# Save the datasets to CSV files
train_df.to_csv('train.csv', index=False)
dev_df.to_csv('dev.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Process completed successfully!")
print(f"Training set samples: {len(train_df)}")
print(f"Development set samples: {len(dev_df)}")
print(f"Test set samples: {len(test_df)}")