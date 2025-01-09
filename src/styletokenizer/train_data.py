import os

from datasets import load_from_disk, Dataset
from tqdm import tqdm
from styletokenizer.utility.custom_logger import log_and_flush

UMICH_TRAIN_DATASET_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/webbook"
UU_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/webbook"
WEBBOOK_COUNT_PER_ROW = 512

UU_MIXED_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/mixed"


def load_train_dataset(word_count=750_000_000, data_path=UMICH_TRAIN_DATASET_PATH, test=False):
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi
    log_and_flush(f"Loading dataset from {data_path}")
    train_data = load_from_disk(data_path)["train"]
    log_and_flush(f"Loaded dataset with {len(train_data)} rows.")

    if not "webbook" in data_path:
        if not os.path.exists(UU_MIXED_TRAIN_DATASET_PATH) or not "mixed" in data_path:
            log_and_flush(f"Using {word_count} words for pre-training.")
            train_data = create_dataset_with_target_word_count(train_data, word_count)
        if "mixed" in data_path and not os.path.exists(UU_MIXED_TRAIN_DATASET_PATH):
            train_data.save_to_disk(UU_MIXED_TRAIN_DATASET_PATH)
        if test:
            train_data = train_data.select(range(256))
    else:
        # for COUNT_PER_ROW get the number of rows to sample for word_count
        nbr_rows = int(word_count // WEBBOOK_COUNT_PER_ROW)
        nbr_rows = min(nbr_rows, len(train_data))
        log_and_flush(f"Using {nbr_rows * WEBBOOK_COUNT_PER_ROW} words for pre-training.")
        # select as many rows as needed to reach the desired train_size, given one row has count COUNT_PER_ROW,
        #   this is not affected by random seed
        if test:
            train_data = train_data.select(range(256))
        else:
            train_data = train_data.select(range(nbr_rows))
    return train_data


def truncate_text(text, max_words=512):
    """Truncate the text to the specified maximum number of words."""
    words = text.split()
    return " ".join(words[:max_words])


def create_dataset_with_target_word_count(dataset, target_word_count):
    """
    Create a new dataset with a cumulative word count that reaches the target.

    Parameters:
        dataset (Dataset): Original Hugging Face dataset with 'text' and 'word_count' columns.
        target_word_count (int): Target total word count for the new dataset.

    Returns:
        Dataset: New dataset with cumulative word count reaching the target.
    """
    # Initialize cumulative word count and new dataset storage
    cumulative_word_count = 0
    new_rows = []

    for row in tqdm(dataset):
        # If word count exceeds 512, truncate the text and adjust the word count
        if row['word_count'] > 512:
            row['text'] = truncate_text(row['text'], max_words=512)
            row['word_count'] = 512

        # Check if adding this row exceeds the target
        if cumulative_word_count + row['word_count'] <= target_word_count:
            new_rows.append(row)
            cumulative_word_count += row['word_count']
        else:
            # Calculate the remaining word count
            remaining_words = target_word_count - cumulative_word_count
            if remaining_words > 0:
                row['text'] = truncate_text(row['text'], max_words=remaining_words)
                row['word_count'] = remaining_words
                new_rows.append(row)
                cumulative_word_count += remaining_words
            log_and_flush(f"Reached target word count of {target_word_count}.")
            break  # Stop adding rows once the target is reached

    # Create a new Dataset
    new_dataset = Dataset.from_dict({key: [row[key] for row in new_rows] for key in new_rows[0].keys()})
    log_and_flush(f"New dataset has {len(new_dataset)} rows and {cumulative_word_count} words.")
    return new_dataset
