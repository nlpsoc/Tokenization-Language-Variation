import argparse
import os

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets import load_dataset
import random


def create_balanced_dataset(total_word_count):
    # Load datasets
    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")

    # Keep only the 'text' and 'id' columns for Wikipedia
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text" and col != "id"])

    # Define the word ratio
    wiki_ratio = 3.125  # Wikipedia to BooksCorpus ratio

    # Calculate target word counts
    wiki_word_count = int(total_word_count * (wiki_ratio / (1 + wiki_ratio)))
    bookcorpus_word_count = total_word_count - wiki_word_count

    print(f"Target word count for Wikipedia: {wiki_word_count}")
    print(f"Target word count for BooksCorpus: {bookcorpus_word_count}")

    # Function to count words in a text
    def count_words(text):
        return len(text.split())

    # Function to sample texts from a dataset to reach a target word count
    def sample_texts(dataset, target_word_count, source_name, use_id=False):
        sampled_texts = []
        current_word_count = 0

        indices = list(range(len(dataset)))
        random.shuffle(indices)  # Shuffle to ensure randomness

        for i in indices:
            text = dataset[i]['text']
            text_word_count = count_words(text)

            if current_word_count + text_word_count <= target_word_count:
                text_entry = {
                    'id': dataset[i]['id'] if use_id else i,  # Use 'id' if available, else use index
                    'text': text,
                    'word_count': text_word_count,
                    'source': source_name
                }
                sampled_texts.append(text_entry)
                current_word_count += text_word_count

            if current_word_count >= target_word_count:
                break

        return sampled_texts, current_word_count

    # Sample texts from each dataset
    sampled_wiki_texts, wiki_actual_word_count = sample_texts(wiki, wiki_word_count, "wikipedia", use_id=True)
    sampled_bookcorpus_texts, bookcorpus_actual_word_count = sample_texts(bookcorpus, bookcorpus_word_count,
                                                                          "bookcorpus")

    # Combine sampled texts into a single list
    combined_texts = sampled_wiki_texts + sampled_bookcorpus_texts

    # Print actual word counts achieved
    print(f"Actual word count from Wikipedia: {wiki_actual_word_count}")
    print(f"Actual word count from BooksCorpus: {bookcorpus_actual_word_count}")

    return combined_texts


def main(word_count=1_000_000_000):
    balanced_dataset = create_balanced_dataset(word_count)
    output_path = "/shared/3/projects/hiatus/EVAL_wegmann/data/train-corpora/wikibook"
    save_to_huggingface_format(balanced_dataset, output_path)

    # Print out some information about the resulting dataset
    print(f"Total word count: {word_count}")
    print(f"Number of examples: {len(balanced_dataset)}")
    print(f"Sample entry: {balanced_dataset[0]}")  # Show the first dictionary entry

    return


if __name__ == '__main__':
    # create command line arguments: word count to sample
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_count", type=int, help="Number of words to sample")
    args = parser.parse_args()
    main(args.word_count)
