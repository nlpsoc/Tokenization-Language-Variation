"""
    scripts to sample from s2orc dataset
    see also: https://github.com/allenai/s2orc/
"""
import os
import json
import random

from styletokenizer.utility.custom_logger import log_and_flush
from utility.datasets_helper import make_text_fit_word_max
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

WORD_COUNT = DOMAIN_WORDCOUNT_DICT["s2orc"]  # 100_000_000


def count_words(text):
    return len(text.split())


def read_files_and_sample(path, target_word_count, test=False):
    total_word_count = 0
    sampled_items = []

    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('s2orc_')]
    num_files = len(files)
    words_per_file = target_word_count // num_files
    words_per_file = max(words_per_file, 1)  # at least 1 word per file for test purposes
    log_and_flush(f"Sampling {words_per_file} words from each of {num_files} files")
    domain_name = "s2orc"

    for file_path in files:
        word_count = 0
        with open(file_path, 'r') as f:
            log_and_flush(f"Reading file: {file_path}")
            lines = f.readlines()
            total_lines = len(lines)
            random_indices = random.sample(range(total_lines), total_lines)  # Shuffle the line indices

            for idx in random_indices:
                if word_count >= words_per_file:
                    break

                line = json.loads(lines[idx])
                corpusid = line.get('corpusid')
                text = line.get('content', {}).get('text', "")
                if not type(text) == str:
                    continue
                text, cur_word_count = make_text_fit_word_max(text)

                sampled_items.append({
                    "id": corpusid,
                    "text": text,
                    "word_count": cur_word_count,
                    "domain": domain_name,
                    "source": domain_name
                })
                word_count += cur_word_count
                total_word_count += cur_word_count
                if test:
                    break

            log_and_flush(f"Sampled word count for file {file_path}: {word_count}")

        if test:
            break
    log_and_flush(f"Total sampled word count: {total_word_count}")

    return sampled_items


def sample_s2orc_texts(required_word_count=WORD_COUNT, test=False):
    sampled_items = read_files_and_sample(s2orc_path, required_word_count, test=test)
    return sampled_items


s2orc_path = "/shared/3/projects/citation-context/s2orc/s2orc"
