"""
    create the wikipedia fitting dataset
"""
import gzip
import random

from fitting_corpora import CORPORA_WIKIPEDIA
from styletokenizer.utility.datasets_helper import save_to_huggingface_format


def count_words_in_line(line):
    return len(line.split())


def extract_random_lines(file_path, target_word_count=1_500_000_000, seed=42):
    random.seed(seed)
    total_word_count = 0
    selected_lines = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        lines = file.readlines()
        random.shuffle(lines)

        for line_number, line in enumerate(lines, 1):
            line_word_count = count_words_in_line(line)
            if total_word_count + line_word_count > target_word_count:
                break
            total_word_count += line_word_count
            selected_lines.append({"id": line_number, "text": line, "word_count": line_word_count})
    return selected_lines


def main():
    # Replace with your actual file path and desired output path
    input_path = "/shared/3/datasets/wikipedia/enwiki/pages-articles/enwiki-20230601-pages-articles.clean-text.txt.gz"
    output_path = CORPORA_WIKIPEDIA

    lines = extract_random_lines(input_path)
    save_to_huggingface_format(lines, output_path)


if __name__ == "__main__":
    main()
