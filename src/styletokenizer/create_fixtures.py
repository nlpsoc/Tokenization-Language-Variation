import pandas as pd
import argparse
import os  # to work with file paths

from styletokenizer.utility.filesystem import get_dir_to_src


def save_first_10_rows_with_prefix(input_filepath):
    """ generated 2024-04-24

    :param input_filepath:
    :return:
    """
    # Load the TSV file
    df = pd.read_csv(input_filepath,  sep="\t\t", header=None)

    # Select the first 10 rows
    first_10_rows = df.head(10)

    # Create the target directory path
    target_dir = os.path.join(get_dir_to_src(), 'test', 'fixtures')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # create the directory if it doesn't exist

    # Get the original file name from the input file path
    file_name = os.path.basename(input_filepath)

    # Create the new file name by prepending "FIXTURE-10" to the original file name
    new_file_name = f"FIXTURE-10-{file_name}"

    # Construct the full output file path in the target directory
    output_filepath = os.path.join(target_dir, new_file_name)

    # Save the first 10 rows to the new TSV file in the target directory
    first_10_rows.to_csv(output_filepath, sep='\t', index=False)

    return output_filepath  # return the new file path for reference


def main(tsv_path):
    save_first_10_rows_with_prefix(tsv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TextClassifier model.')
    parser.add_argument('-tsv_path', metavar='tsv_path', type=str,
                        default=(get_dir_to_src() + "/../data/2021_jian_idiolect/Amazon/Amazon_test_contrastive"),
                        help='the path to the training data file')
    args = parser.parse_args()
    main(tsv_path=args.tsv_path)
