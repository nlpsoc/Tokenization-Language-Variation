import os
import json
import random

s2orc_path = "/shared/3/projects/citation-context/s2orc/s2orc"

WORD_COUNT = 100_000_000


def count_words(text):
    return len(text.split())


def read_files_and_sample(path, target_word_count):
    total_word_count = 0
    sampled_items = []
    sampled_ids = []
    sampled_texts = []
    samled_word_counts = []

    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('s2orc_')]
    num_files = len(files)
    words_per_file = target_word_count // num_files
    words_per_file = max(words_per_file, 1)  # at least 1 word per file for test purposes
    print(f"Sampling {words_per_file} words from each of {num_files} files")

    for file_path in files:
        word_count = 0
        with open(file_path, 'r') as f:
            print(f"Reading file: {file_path}")
            lines = f.readlines()
            total_lines = len(lines)
            random_indices = random.sample(range(total_lines), total_lines)  # Shuffle the line indices

            for idx in random_indices:
                if word_count >= words_per_file:
                    break

                line = json.loads(lines[idx])
                corpusid = line.get('corpusid')
                text = line.get('content', {}).get('text', "")
                words = text.split()

                sampled_items.append({
                    "id": corpusid,
                    "text": text,
                    "word_count": len(words),
                    "domain": "s2orc",
                    "source": "s2orc"
                })
                # sampled_texts.append(text)
                # sampled_ids.append(corpusid)
                # samled_word_counts.append(len(words))
                word_count += len(words)
                total_word_count += len(words)
            print(f"Sampled word count for file {file_path}: {word_count}")

    return sampled_items


def sample_s2orc_texts(required_word_count=WORD_COUNT):
    sampled_items = read_files_and_sample(s2orc_path, required_word_count)
    return sampled_items

    # {
    #     "id": sampled_ids,  # corpusid as used in original s2orc dataset
    #     "domain": ["s2orc"] * len(sampled_texts),
    #     "source": ["s2orc"] * len(sampled_texts),
    #     "word_count": sampled_wc,
    #     "text": sampled_texts,
    # }
