import os
import re
import json
import csv


def gather_data(folder_path):
    """
    Go through the folder looking for files named 'problem-X.txt'.
    For each found file, look for 'truth-problem-X.json'.
    Extract paragraph pairs and their corresponding Author Change labels,
    and return a list of [paragraph_1, paragraph_2, author_change].
    """
    # A regex to capture the problem ID in 'problem-X.txt'
    problem_txt_pattern = re.compile(r'^problem-(.+)\.txt$')

    # We'll store all rows (text_1, text_2, author_change) in one list
    all_rows = []

    # List all files in the folder
    folder_files = os.listdir(folder_path)

    for filename in folder_files:
        match = problem_txt_pattern.match(filename)
        if match:
            # Extract "X" from 'problem-X.txt' (e.g., "1", "2", "10")
            problem_id = match.group(1)

            # Build the expected filenames
            txt_path = os.path.join(folder_path, f"problem-{problem_id}.txt")
            json_path = os.path.join(folder_path, f"truth-problem-{problem_id}.json")

            if problem_id == "244":
                print(f"txt_path: {txt_path}")

            # Skip if the corresponding JSON doesn't exist
            if not os.path.exists(json_path):
                print(f"WARNING: No matching JSON for {txt_path}")
                continue

            # --- Read paragraphs from the TXT file ---
            with open(txt_path, "r", newline="") as f_txt:
                raw_text = f_txt.read()
                # Split on double newlines (adjust as needed)
                paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]

            # --- Read the 'changes' list from the JSON file ---
            with open(json_path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)
                changes = data["changes"]  # expecting a list

            # --- Gather the data into rows ---
            # For each consecutive pair of paragraphs (i, i+1)
            for i in range(len(paragraphs) - 1):
                p1 = paragraphs[i]
                p2 = paragraphs[i + 1]
                if i >= len(changes):
                    print(f"Ran out of changes for {txt_path} for pair {i}-{i+1}")
                author_change_label = changes[i]

                all_rows.append([p1, p2, author_change_label])

    return all_rows


def write_combined_csv(rows, output_csv_path):
    """
    Given a list of rows where each row = [text_1, text_2, author_change],
    write them all to a single CSV with a header.
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["text 1", "text 2", "Author Change"])
        # Write the data rows
        writer.writerows(rows)

    # make pandas dataframe
    import pandas as pd
    df = pd.read_csv(output_csv_path)
    # distribution of Author Change
    print(df['Author Change'].value_counts())



def main():
    # 1. Specify the folder containing 'problem-X.txt' and 'truth-problem-X.json' files
    folder_path = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/PAN/hard/train"
    # 2. Specify the single CSV filename you want
    output_csv_filename = "PAN-hard_train.csv"
    # 3. Gather rows from all matching text/JSON pairs
    rows = gather_data(folder_path)
    # 4. Write them out to one CSV file
    write_combined_csv(rows, os.path.join(folder_path, output_csv_filename))

    folder_path = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/PAN/hard/validation"
    # 2. Specify the single CSV filename you want
    output_csv_filename = "PAN-hard_validation.csv"
    # 3. Gather rows from all matching text/JSON pairs
    rows = gather_data(folder_path)
    # 4. Write them out to one CSV file
    write_combined_csv(rows, os.path.join(folder_path, output_csv_filename))

    print(f"Done! Output CSV created at: {os.path.join(folder_path, output_csv_filename)}")


if __name__ == "__main__":
    main()