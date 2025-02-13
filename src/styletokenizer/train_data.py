"""
    loading of train dataset for BERT training
"""
import os

from datasets import load_from_disk, Dataset
from tqdm import tqdm

from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT, WORD_COUNT_TOTAL
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.env_variables import at_uu, at_local
from styletokenizer.fitting_corpora import CORPORA_MIXED
import re

TEST_ROWS = 1048 * 2

UMICH_TRAIN_DATASET_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/webbook"
UU_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/webbook"
COUNT_PER_ROW = 512

UU_MIXED_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/mixed"
LOCAL_MIXED_TRAIN_DATASET_PATH = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/train-corpora/mixed"


def load_train_dataset(word_count=750_000_000, data_path=UMICH_TRAIN_DATASET_PATH, test=False):
    assert "webbook" in data_path or "mixed" in data_path, "Only webbook and mixed datasets are supported."
    assert at_local() or at_uu(), "Only local and UU paths are supported."
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi

    if "mixed" in data_path:
        if data_path != CORPORA_MIXED:
            raise ValueError("Expecting the path to the original fitting corpus here to use it's val and test splits.")
        if not os.path.exists(UU_MIXED_TRAIN_DATASET_PATH):
            log_and_flush(f"Using {word_count} words for pre-training.")
            train_data = load_from_disk(data_path)
            train_data = create_dataset_with_fixed_row_length(train_data, word_count)
            train_data.save_to_disk(UU_MIXED_TRAIN_DATASET_PATH)
        else:
            log_and_flush(f"Loading dataset from {UU_MIXED_TRAIN_DATASET_PATH}")
            train_data = load_from_disk(UU_MIXED_TRAIN_DATASET_PATH)
            log_and_flush(f"Loaded dataset with {len(train_data)} rows.")
    else:
        log_and_flush(f"Loading dataset from {data_path}")
        train_data = load_from_disk(data_path)["train"]
        log_and_flush(f"Loaded dataset with {len(train_data)} rows.")

    # for COUNT_PER_ROW get the number of rows to sample for word_count
    nbr_rows = int(word_count // COUNT_PER_ROW)
    nbr_rows = min(nbr_rows, len(train_data))
    log_and_flush(f"Using {nbr_rows * COUNT_PER_ROW} words for pre-training.")
    # select as many rows as needed to reach the desired train_size, given one row has count COUNT_PER_ROW,
    #   this is not affected by random seed
    if test:
        train_data = train_data.select(range(TEST_ROWS))
    else:
        train_data = train_data.select(range(nbr_rows))
    return train_data


def get_end_position_n_word(input_string, n):
    # Match all words in the string
    word_matches = list(re.finditer(r'\S+', input_string))

    # If there are fewer than n words, return the whole string
    if len(word_matches) < n:
        return -1

    # Get the end position of the nth word
    end_position = word_matches[n - 1].end()

    # Return the substring up to that position
    return end_position

def truncate_or_merge_text_preserving_whitespace(rows, leftover_text="", max_words=512):
    """
    Merge or truncate rows to create a single text entry of exactly max_words,
    preserving original whitespacing by using character positions.
    """
    combined_text = leftover_text
    current_word_count = len(combined_text.split())
    remaining_text = ""

    if current_word_count >= max_words:
        char_position = get_end_position_n_word(combined_text, max_words)
        remaining_text = combined_text[char_position:]
        combined_text = combined_text[:char_position]
        return combined_text, max_words, remaining_text

    for row in rows:
        remaining_space = max_words - current_word_count
        if row["word_count"] <= remaining_space:
            if len(combined_text) > 0:
                combined_text += "\n"
            combined_text += row["text"]
            current_word_count += row["word_count"]
        else:
            # Find the character position for the remaining words
            text = row["text"]
            char_position = get_end_position_n_word(text, remaining_space)
            combined_text += "\n" + text[:char_position]
            current_word_count += remaining_space
            remaining_text = text[char_position:]
            break  # Once we reach max_words, stop combining rows

    return combined_text, current_word_count, remaining_text  # Return combined text and fixed word count


def create_dataset_with_fixed_row_length(dataset, target_word_count):
    """
    Create a new dataset where each row is exactly 512 words by merging rows
    within the same domain, preserving original whitespacing.
    """
    # words per domain
    relative_contribution_per_domain = {domain: int((word_count / WORD_COUNT_TOTAL)*target_word_count)
                                        for domain, word_count in DOMAIN_WORDCOUNT_DICT.items()}

    # Group rows by domain
    grouped_data = {}
    for row in tqdm(dataset, "Grouping Domains"):
        domain = row["domain"]
        if domain not in grouped_data:
            grouped_data[domain] = []
        grouped_data[domain].append(row)

    new_rows = []
    cumulative_word_count = 0

    # assert that the domain keys in relative_contribution_per_domain are the same keys as in grouped_data
    if not set(relative_contribution_per_domain.keys()) == set(grouped_data.keys()):
        print(set(relative_contribution_per_domain.keys()))
        print(set(grouped_data.keys()))
        raise ValueError("The keys in relative_contribution_per_domain must match the keys in grouped")

    # Process each domain group
    for domain, rows in tqdm(grouped_data.items(), "Domains"):
        temp_rows = []
        domain_word_count = 0
        remaining_text = ""
        for row in tqdm(rows, f"Rows in {domain}"):
            temp_rows.append(row)

            # If the cumulative word count of temp_rows reaches 512, combine them
            total_words = sum(r["word_count"] for r in temp_rows) + len(remaining_text.split())
            while total_words >= 512:
                combined_text, final_word_count, remaining_text = (
                    truncate_or_merge_text_preserving_whitespace(temp_rows, remaining_text, max_words=512))
                new_rows.append({"domain": domain, "text": combined_text, "word_count": final_word_count})
                domain_word_count += final_word_count
                temp_rows = []  # Reset temp_rows
                total_words = len(remaining_text.split())  # Update total words

            # Stop adding rows once the target is reached
            if domain_word_count >= relative_contribution_per_domain[domain]:
                cumulative_word_count += domain_word_count
                break

        if cumulative_word_count >= target_word_count:
            break

    log_and_flush(f"Created a new dataset with {cumulative_word_count} words.")
    # Create a new Dataset
    new_dataset = Dataset.from_dict({key: [row[key] for row in new_rows] for key in new_rows[0].keys()})
    # shuffle the new dataset
    new_dataset = new_dataset.shuffle(seed=42)
    return new_dataset
