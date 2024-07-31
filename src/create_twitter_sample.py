import bz2
import json
import os
from datasets import Dataset, DatasetDict

from styletokenizer.utility.datasets_helper import save_to_huggingface_format


def count_words(text):
    return len(text.split())


def process_file(file_path, target_word_count_per_file, data):
    cumulative_word_count = 0
    with bz2.open(file_path, 'rt') as file:
        for line in file:
            tweet = json.loads(line)
            text = tweet.get("text", "")
            tweet_id = tweet.get("id", "")
            word_count = count_words(text)

            cumulative_word_count += word_count
            data.append({"id": tweet_id, "text": text, "word_count": word_count})

            if cumulative_word_count >= target_word_count_per_file:
                return data, cumulative_word_count
    return data, cumulative_word_count


def sample_texts_from_files(directory, target_word_count):
    """
        get the number of compressed files and determine how much to sample from each file,
        st target word count is distributed equally across bz2 files
    """
    files = [os.path.join(root, file_name)
             for root, _, files in os.walk(directory)
             for file_name in sorted(files)
             if not "2022" in file_name and (file_name.endswith('p2.bz2') or file_name.endswith('p1.bz2'))]
    print(files)
    # testing opening each file
    stream_error = False
    error_files = []
    for file_path in files:
        try:
            with bz2.open(file_path, 'rt') as file:
                for line in file:
                    json.loads(line)
                    break
        except Exception as e:
            print(f"Error opening file: {file_path}")
            print(e)
            error_files.append(file_path)
            stream_error = True
    if stream_error:
        files = [file for file in files if file not in error_files]
        print(f"Error opening {len(error_files)} files. Continuing with {len(files)} files.")

    num_files = len(files)
    target_word_count_per_file = target_word_count // num_files
    data = []

    for filen_index, file_path in enumerate(files):
        print(f"At file {filen_index}/{num_files} called {file_path}")
        print(f"Extracting {target_word_count_per_file} words")
        data, _ = process_file(file_path, target_word_count_per_file, data)

    return data


def main():
    directory = '/nfs/locker/twitter-decahose-locker/2021'
    output_path = '/shared/3/projects/hiatus/TOKENIZER_wegmann/data/fitting-corpora/twitter'
    target_word_count = 1_500_000_000
    sampled_twitter_data = sample_texts_from_files(directory, target_word_count)
    save_to_huggingface_format(sampled_twitter_data, output_path)

# dataset = Dataset.from_dict({"id": [item["id"] for item in data],
#                              "text": [item["text"] for item in data],
#                              "word_count": [item["word_count"] for item in data]})
# dataset.save_to_disk(output_path)
