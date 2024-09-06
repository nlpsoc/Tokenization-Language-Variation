import argparse
import bz2
import json
import os
import random

from styletokenizer.fitting_corpora import CORPORA_TWITTER
from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from styletokenizer.utility.custom_logger import log_and_flush


def count_words(text):
    return len(text.split())


def process_file(file_path, target_word_count_per_file, data):
    cumulative_word_count = 0
    appended_texts = set()  # To store hashes of texts that have already been appended

    # Open and process the file line by line
    with bz2.open(file_path, 'rt') as file:
        for line in file:
            tweet = json.loads(line)
            lang = tweet.get("lang", "")
            if lang != "en":  # Only consider English tweets
                continue
            text = tweet.get("text", "")
            # test if text starts with "RT" and skip if it does
            if text.startswith("RT"):  # is a retweet
                continue
            tweet_id = tweet.get("id", "")
            text_hash = hash(text)  # Create a hash of the text for quicker lookups

            # Only append if hash of text is unique
            if text_hash not in appended_texts:
                appended_texts.add(text_hash)  # Add the hash, not the whole text
                word_count = count_words(text)
                cumulative_word_count += word_count
                data.append({"id": tweet_id, "text": text, "word_count": word_count})

            # Stop if the cumulative word count has reached the target
            if cumulative_word_count >= target_word_count_per_file:
                return data, cumulative_word_count

    return data, cumulative_word_count


def sample_texts_from_files(directory, target_word_count):
    """
    Get the number of compressed files and determine how much to sample from each file,
    so that the target word count is distributed equally across bz2 files.
    """
    # Gather all relevant bz2 files (p1 and p2), excluding those with "2022" in their names
    files = [os.path.join(root, file_name)
             for root, _, files in os.walk(directory)
             for file_name in sorted(files)
             if not "2022" in file_name and (file_name.endswith('p2.bz2') or file_name.endswith('p1.bz2'))]
    log_and_flush(f"Found files: {files}")

    # remove files with sizes < 10GB
    files = [file for file in files if os.path.getsize(file) > 10_000_000_000]
    log_and_flush(f"Files with size > 10GB: {files}")

    # Test opening each file
    stream_error = False
    error_files = []

    for file_path in files:
        try:
            with bz2.open(file_path, 'rt') as file:
                for line in file:
                    json.loads(line)
                    break
        except Exception as e:
            log_and_flush(f"Error opening file: {file_path}")
            log_and_flush(e)
            error_files.append(file_path)
            stream_error = True
    if stream_error:
        files = [file for file in files if file not in error_files]
        log_and_flush(f"Error opening {len(error_files)} files. Continuing with {len(files)} files.")

    num_files = len(files)
    target_word_count_per_file = target_word_count // num_files
    target_word_count_per_file = max(target_word_count_per_file, 1)  # at least 1 word even for testing per file
    data = []

    for filen_index, file_path in enumerate(files):
        log_and_flush(f"At file {filen_index}/{num_files} called {file_path}")
        log_and_flush(f"Extracting {target_word_count_per_file} words")
        data, _ = process_file(file_path, target_word_count_per_file, data)

    return data


def main(test=False):
    directory = '/nfs/locker/twitter-decahose-locker/2021'
    output_path = CORPORA_TWITTER
    if test:
        target_word_count = 10
        output_path += "_test"
    else:
        target_word_count = 1_500_000_000
    log_and_flush(f"Sampling {target_word_count} words from Twitter data")
    sampled_twitter_data = sample_texts_from_files(directory, target_word_count)
    # shuffle list
    import random
    random.seed(42)
    random.shuffle(sampled_twitter_data)
    save_to_huggingface_format(sampled_twitter_data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run the script in test mode')
    args = parser.parse_args()
    main(test=args.test)


