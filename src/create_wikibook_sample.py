import argparse
import os
import re



cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from styletokenizer.utility.custom_logger import log_and_flush
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets import load_dataset
import random
from pathlib import Path

COUNT_PER_ROW = 512


def sample_texts_from_wiki_dataset(target_word_count, source_name, use_id=False):
    log_and_flush("Sampling from Wikipedia")
    # Load datasets
    dataset = load_dataset("wikipedia", "20220301.en", split="train")

    # Keep only the 'text' and 'id' columns for Wikipedia
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text" and col != "id"])

    sampled_texts = []
    current_word_count = 0

    indices = list(range(len(dataset)))
    random.shuffle(indices)  # Shuffle to ensure randomness

    for i in indices:
        text = dataset[i]['text']
        # Match words and preserve whitespaces
        #   divides string in all consecutive non-whitespace characters and all whitespace characters
        tokens = re.findall(r'\S+|\s+', text)
        text_word_count = int(len(tokens) / 2)

        if (current_word_count < target_word_count) and text_word_count >= COUNT_PER_ROW:
            text_entry = {
                'id': dataset[i]['id'] if use_id else i,  # Use 'id' if available, else use index
                'text': ''.join(tokens[:COUNT_PER_ROW * 2]),
                'word_count': COUNT_PER_ROW,
                'source': source_name
            }
            sampled_texts.append(text_entry)
            current_word_count += text_word_count

        if current_word_count >= target_word_count:
            break

    log_and_flush(f"Extracted {current_word_count} words")

    return sampled_texts, current_word_count


def sample_texts_from_bookcorpus_dataset(target_word_count, source_name, use_id=False, test=False):
    bookcorpus_path = '/shared/2/datasets/gpt3-books/books3/the-eye.eu/public/Books/Bibliotik/'
    base_path = Path(bookcorpus_path)
    subfolders = [f for f in base_path.iterdir() if f.is_dir() and f.name != '0_Other']
    # get all text files with path
    bookcorpus_files = []
    for folder in subfolders:
        bookcorpus_files.extend(list(folder.glob('**/*.txt')))
    if test:
        bookcorpus_files = bookcorpus_files[:5]
    word_per_file = target_word_count / len(bookcorpus_files)
    # determine how many excerpts to extract
    num_excerpts = word_per_file // COUNT_PER_ROW

    sampled_texts = []
    current_word_count = 0
    # caluculate the number of words to extract from each file
    for file in bookcorpus_files:
        log_and_flush(f"At file {file}")
        with open(file, 'r') as f:
            text = f.read()

        tokens = re.findall(r'\S+|\s+', text)

        # get num_excerpts random starting points that do not overlap in 512 token chunks
        possible_starts = list(range(0, len(tokens) - COUNT_PER_ROW * 2, COUNT_PER_ROW * 2))
        random.shuffle(possible_starts)

        for start in possible_starts[:num_excerpts]:
            text_entry = {
                'id': file.name if use_id else file,
                'text': ''.join(tokens[start:start + COUNT_PER_ROW * 2]),
                'word_count': COUNT_PER_ROW,
                'source': source_name
            }
            sampled_texts.append(text_entry)
            current_word_count += COUNT_PER_ROW
        log_and_flush(f"Extracted {current_word_count} words")
        return sampled_texts, current_word_count


def create_balanced_dataset(total_word_count, test=False):
    if test:
        total_word_count = 512*10
    # Define the word ratio
    wiki_ratio = 3.125  # Wikipedia to BooksCorpus ratio

    # Calculate target word counts
    wiki_word_count = int(total_word_count * (wiki_ratio / (1 + wiki_ratio)))
    bookcorpus_word_count = total_word_count - wiki_word_count

    log_and_flush(f"Target word count for Wikipedia: {wiki_word_count}")
    log_and_flush(f"Target word count for BooksCorpus: {bookcorpus_word_count}")

    random.seed(42)

    # Sample texts from each dataset
    sampled_bookcorpus_texts, bookcorpus_actual_word_count = (
        sample_texts_from_bookcorpus_dataset(bookcorpus_word_count, "bookcorpus", use_id=True, test=test))
    sampled_wiki_texts, wiki_actual_word_count = sample_texts_from_wiki_dataset(wiki_word_count, "wikipedia",
                                                                                use_id=True)

    # Combine sampled texts into a single list
    combined_texts = sampled_wiki_texts + sampled_bookcorpus_texts

    # shuffle the list of dicts

    random.shuffle(combined_texts)

    # Print actual word counts achieved
    print(f"Actual word count from Wikipedia: {wiki_actual_word_count}")
    print(f"Actual word count from BooksCorpus: {bookcorpus_actual_word_count}")

    return combined_texts


def main(word_count=3_300_000_000, test=False):
    balanced_dataset = create_balanced_dataset(word_count, test=test)
    output_path = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/wikibook"
    if test:
        output_path += "_test"
    save_to_huggingface_format(balanced_dataset, output_path)

    # Print out some information about the resulting dataset
    print(f"Total word count: {word_count}")
    print(f"Number of examples: {len(balanced_dataset)}")
    print(f"Sample entry: {balanced_dataset[0]}")  # Show the first dictionary entry

    return


if __name__ == '__main__':
    # create command line arguments: word count to sample
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the script in test mode")
    args = parser.parse_args()
    main(test=args.test)
