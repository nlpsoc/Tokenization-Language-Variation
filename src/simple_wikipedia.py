import argparse
import random

import pandas as pd
import os


def extract_third_column(line, file_path, line_num):
    """
    Extracts the third column from a tab-separated line.
    Raises an error if the line doesn't have at least three columns.
    """
    parts = line.strip().split('\t')
    if len(parts) < 3:
        raise ValueError(f"Line {line_num} in {file_path} does not have at least three columns.")
    return parts[2]


def main(normal_path, simple_path, output_dir):
    """
    Reads two text files line-by-line in parallel, keeps lines
    that differ, labels them, shuffles and splits into train/dev/test,
    then writes the splits to CSV files.
    """

    normal_df = pd.read_csv(normal_path, sep='\t', header=None)
    simple_df = pd.read_csv(simple_path, sep='\t', header=None)

    # Check that both files have the same number of lines
    if len(normal_df) != len(simple_df):
        raise ValueError("Files must have the same number of lines.")

    # Collect distinct lines (parallel positions)
    texts = {
        "normal": [],
        "simple": []
    }
    for normal_line, simple_line in zip(normal_df[2], simple_df[2]):
        if normal_line != simple_line:
            # Add each distinct line to our data
            texts["normal"].append(normal_line)
            texts["simple"].append(simple_line)

    # select 50k indices from the normal text
    total_len = len(texts["normal"])
    selected_indices = random.sample(list(range(total_len)), 50000)

    # Create a pandas DataFrame
    df = pd.DataFrame({"text": [texts["normal"][i] for i in selected_indices] +
                               [texts["simple"][i] for i in selected_indices],
                       "label": ["normal"] * len(selected_indices) + ["simple"] * len(selected_indices)
                       })

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train/dev/test
    # 80% train, 10% dev, 10% test
    train_split = 0.7
    dev_split = 0.15

    total_len = len(df)
    train_end = int(train_split * total_len)
    dev_end = int((train_split + dev_split) * total_len)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    dev_df = df.iloc[train_end:dev_end].reset_index(drop=True)
    test_df = df.iloc[dev_end:].reset_index(drop=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    train_file = os.path.join(output_dir, "train.csv")
    dev_file = os.path.join(output_dir, "dev.csv")
    test_file = os.path.join(output_dir, "test.csv")

    # number of lines in train, dev, test
    print(f"Train: {len(train_df)} lines")
    print(f"Dev: {len(dev_df)} lines")
    print(f"Test: {len(test_df)} lines")

    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Data saved:\n  Train -> {train_file}\n  Dev   -> {dev_file}\n  Test  -> {test_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Read two text files line-by-line in parallel, keep distinct lines, label them, shuffle/split, and save to CSV."
    # )
    # parser.add_argument("--normal", help="Path to the normal wiki text file.")
    # parser.add_argument("--simple", help="Path to the simple text file.")
    #
    # # get output directory from normal
    # args = parser.parse_args()
    # output_dir = os.path.dirname(args.normal)
    # #
    # # parser.add_argument("-o", "--output_dir", default="output_data",
    # #                     help="Directory to save the train/dev/test CSV files.")
    #
    # main(args.normal, args.normal, output_dir)

    main(
        "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/simplification/sentence-aligned.v2/normal.aligned",
        "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/simplification/sentence-aligned.v2/simple.aligned",
        "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/simplification/sentence-aligned.v2")
# python simple_wikipedia.py --normal "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/simplification/sentence-aligned.v2/normal.aligned" --simple "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/simplification/sentence-aligned.v2/simple.aligned"
