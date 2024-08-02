from datasets import Dataset, DatasetDict, load_from_disk


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
        keys = data[0].keys()
        dataset_dict = {key: [item[key] for item in data] for key in keys}
        return Dataset.from_dict(dataset_dict)

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
    print(f"Saved dataset to {output_path}")

    # print statistics
    print(f"Total word count: {total_word_count}")
    print(f"Train word count: {sum(train_dataset['word_count'])}")
    print(f"Dev word count: {sum(dev_dataset['word_count'])}")
    print(f"Test word count: {sum(test_dataset['word_count'])}")

    # distribution over word count over "domain" for each split if "domain" was provided
    if "domain" in train_dataset[0]:
        for split in [train_dataset, dev_dataset, test_dataset]:
            domain_word_count = {}
            for domain, word_count in zip(split["domain"], split["word_count"]):
                if domain not in domain_word_count:
                    domain_word_count[domain] = 0
                domain_word_count[domain] += word_count
            print(f"Domain word count distribution for {split}: {domain_word_count}")


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


def train_text_generator(dataset_path):
    """
    Generator that yields text data from a Huggingface-formatted dataset.

    Args:
    - dataset_path (str): Path to the Huggingface-formatted dataset.

    Yields:
    - text (str): Text data from the dataset.
    """
    for data_entry in huggingface_format_generator(dataset_path):
        yield data_entry["text"]


# def batch_text_generator(dataset_path, split="train", batch_size=1000):
#     """
#     Generator that yields data entries from a Huggingface-formatted dataset in batches.
#
#     Args:
#     - dataset_path (str): Path to the Huggingface-formatted dataset.
#     - split (str): The split of the dataset to use (default is "train").
#     - batch_size (int): Number of data entries in each batch (default is 1000).
#
#     Yields:
#     - batch (list of dict): A batch of data entries from the dataset.
#     """
#     dataset = Dataset.load_from_disk(dataset_path)
#     batch = []
#     for i in range(len(dataset[split])):
#         batch.append(dataset[split][i]["text"])
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#
#     # Yield the last batch if it's not empty and has less than batch_size elements
#     if batch:
#         yield batch
