import os
import json
import random

s2orc_path = "/shared/3/projects/citation-context/s2orc/s2orc"

WORD_COUNT = 100_000_000

def count_words(text):
    return len(text.split())


def read_files_and_sample(path, target_word_count):
    word_count = 0
    sampled_ids = []
    sampled_texts = []

    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('s2orc_')]
    num_files = len(files)
    words_per_file = target_word_count // num_files

    for file_path in files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            total_lines = len(lines)
            random_indices = random.sample(range(total_lines), total_lines)  # Shuffle the line indices

            for idx in random_indices:
                if word_count >= target_word_count:
                    break

                line = json.loads(lines[idx])
                corpusid = line.get('corpusid')
                text = line.get('content', {}).get('text', "")
                words = text.split()

                if word_count + len(words) > target_word_count:
                    remaining_words = target_word_count - word_count
                    sampled_texts.append(" ".join(words[:remaining_words]))
                    sampled_ids.append(corpusid)
                    word_count += remaining_words
                    break
                else:
                    sampled_texts.append(text)
                    sampled_ids.append(corpusid)
                    word_count += len(words)

                if word_count >= words_per_file * (files.index(file_path) + 1):
                    break

        if word_count >= target_word_count:
            break

    return sampled_ids, sampled_texts


def sample_s2orc_texts(required_word_count=WORD_COUNT):
    sampled_ids, sampled_texts = read_files_and_sample(s2orc_path, required_word_count)
    return {
        "id": sampled_ids,  # corpusid as used in original s2orc dataset
        "domain": ["s2orc"] * len(sampled_texts),
        "source": ["s2orc"] * len(sampled_texts),
        "text": sampled_texts,
    }
