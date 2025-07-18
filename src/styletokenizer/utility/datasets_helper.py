"""
    loading the different eval datasets in all different forms (from huggingface datasets, to .csv files)
"""
import os
import re

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import pyarrow as pa

from styletokenizer.utility.custom_logger import log_and_flush


def save_to_huggingface_format(data, output_path, dev_size=0.01, test_size=0.01):
    """
    Saves data to Huggingface format with train/dev/test split based on word counts.

    Args:
    - data (list): List of data entries, each a dict {"id": id, "text": text, "word_count": word_count}.
    - output_path (str): Path to save the dataset.
    - dev_size (float): Proportion of data to be used as dev set.
    - test_size (float): Proportion of data to be used as test set.
    """
    # Calculate total word count
    total_word_count = sum(item["word_count"] for item in data)

    # Calculate target word counts for dev and test sets
    dev_word_count_target = total_word_count * dev_size
    test_word_count_target = total_word_count * test_size

    # Split the data based on the word counts
    def split_data(data, target_word_count):
        split_data = []
        current_word_count = 0
        for item in data:
            split_data.append(item)
            current_word_count += item["word_count"]
            if current_word_count >= target_word_count:
                break
        remaining_data = data[len(split_data):]
        return split_data, remaining_data

    dev_data, remaining_data = split_data(data, dev_word_count_target)
    test_data, train_data = split_data(remaining_data, test_word_count_target)

    # Create datasets for each split
    def create_dataset(data):
        # get all keys that can appear (should be the same for all data entries,
        #   but there might be some issues with domain/source)
        keys = data[0].keys()
        # Create the dataset dictionary with inline type conversion for 'id'
        dataset_dict = {
            key: [str(item[key]) if key == 'id' else item.get(key, "") for item in data]
            for key in keys
        }
        try:
            return Dataset.from_dict(dataset_dict)
        except pa.lib.ArrowInvalid as e:
            log_and_flush(f"Some values have unexpected type: {e}")
            raise ValueError("Inconsistent data to create the dataset")

    train_dataset = create_dataset(train_data)
    dev_dataset = create_dataset(dev_data)
    test_dataset = create_dataset(test_data)

    # Create a DatasetDict to hold the splits
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset
    })

    # Save the dataset to the specified path
    dataset_dict.save_to_disk(output_path)
    log_and_flush(f"Saved dataset to {output_path}")

    # print statistics
    log_and_flush(f"Total word count: {total_word_count}")
    log_and_flush(f"Train word count: {sum(train_dataset['word_count'])}")
    log_and_flush(f"Dev word count: {sum(dev_dataset['word_count'])}")
    log_and_flush(f"Test word count: {sum(test_dataset['word_count'])}")

    # distribution over word count over "domain" for each split if "domain" was provided
    if "domain" in train_dataset[0]:
        for split in [train_dataset, dev_dataset, test_dataset]:
            domain_word_count = {}
            for domain, word_count in zip(split["domain"], split["word_count"]):
                if domain not in domain_word_count:
                    domain_word_count[domain] = 0
                domain_word_count[domain] += word_count
            log_and_flush(f"Domain word count distribution for {split}: {domain_word_count}")


def huggingface_format_generator(dataset_path, split="train"):
    """
    Generator that yields data entries from a Huggingface-formatted dataset.

    Args:
    - dataset_path (str): Path to the Huggingface-formatted dataset.

    Yields:
    - data_entry (dict): A data entry from the dataset.
    """
    dataset = load_from_disk(dataset_path)
    for i in range(len(dataset[split])):
        yield dataset[split][i]


def train_text_generator(dataset_path, split="train"):
    """
    Generator that yields text data from a Huggingface-formatted dataset.

    Args:
    - dataset_path (str): Path to the Huggingface-formatted dataset.

    Yields:
    - text (str): Text data from the dataset.
    """
    for data_entry in huggingface_format_generator(dataset_path, split=split):
        yield data_entry["text"]


def efficient_split_generator(dataset_path, split="dev"):
    data_split = load_from_disk(os.path.join(dataset_path, split))
    for i in range(len(data_split)):
        yield data_split[i]["text"]


def load_data(task_name_or_hfpath=None, csv_file=None, split=None):
    """
        loading the different eval datasets in all different forms (from huggingface datasets,
         locally with datasets, or csv)
    :param split:
    :param task_name_or_hfpath:
    :param csv_file:
    :return:
    """
    # load dev set of the datasets
    if task_name_or_hfpath is not None:
        if os.path.exists(task_name_or_hfpath):
            if split:
                raw_datasets = load_from_disk(os.path.join(task_name_or_hfpath, split))
            else:
                raw_datasets = load_from_disk(task_name_or_hfpath)
            # set task name to last folder in path
            task_name_or_hfpath = os.path.basename(os.path.normpath(task_name_or_hfpath))
        else:
            if task_name_or_hfpath == "snli":
                # SNLI
                raw_datasets = load_dataset(
                    "snli"
                )
                raw_datasets["train"] = raw_datasets["train"].filter(lambda x: x['label'] != -1)
                raw_datasets["validation"] = raw_datasets["validation"].filter(lambda x: x['label'] != -1)
            else:
                # GLUE: MRPC, CoLA, SST-2, QNLI, QQP, RTE, WNLI, STS-B
                raw_datasets = load_dataset(
                    "nyu-mll/glue",
                    task_name_or_hfpath
                )
        log_and_flush(f"Dataset loaded: {raw_datasets}")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"validation": csv_file}  # should also work with list of csv files
        csv = False
        tsv = False
        if type(csv_file) == list:
            if csv_file[0].endswith(".csv"):
                csv = True
            elif csv_file[0].endswith(".tsv"):
                tsv = True
        else:
            if csv_file.endswith(".csv"):
                csv = True
            elif csv_file.endswith(".tsv"):
                tsv = True

        if csv:
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files
            )
        elif tsv:
            # Loading a dataset from local tsv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                delimiter="\t"
            )
        else:
            raise ValueError("Please provide a valid csv or tsv file")

    return raw_datasets


MAX_WORD_COUNT = 512 * 4


def make_text_fit_word_max(text, max_word_count=MAX_WORD_COUNT):
    """
    Cut off text to fit the maximum word count
    :param max_word_count:
    :param text: text to be cut off
    :return: cut off text
    """
    tokens = re.findall(r'\S+|\s+', text)
    text_word_count = int(len(tokens) / 2)

    if text_word_count > max_word_count:
        text = ''.join(tokens[:max_word_count * 2])
        text_word_count = max_word_count
    return text, text_word_count
