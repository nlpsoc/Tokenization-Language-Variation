import argparse
import os
import re

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

from styletokenizer.utility.datasets_helper import save_to_huggingface_format
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.bookcorpus import BOOK3CORPUS_PATH
from datasets import load_dataset
import random
from pathlib import Path
import styletokenizer.utility.the_pile as the_pile

COUNT_PER_ROW = 512


def sample_texts_from_webtext_dataset(target_word_count):
    log_and_flush("Sampling from The Pile's OpenWebText")

    sampled_texts = the_pile.sample_pile_texts(['OpenWebText2'], [target_word_count],
                                               individual_text_length=COUNT_PER_ROW, ensure_en=True)
    current_word_count = sum([entry['word_count'] for entry in sampled_texts])
    log_and_flush(f"Extracted {current_word_count} words")

    return sampled_texts, current_word_count


def sample_texts_from_bookcorpus_dataset(target_word_count, test=False):
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException
    bookcorpus_path = BOOK3CORPUS_PATH
    base_path = Path(bookcorpus_path)
    subfolders = [f for f in base_path.iterdir() if f.is_dir() and f.name != '0_Other']
    # get all text files with path
    bookcorpus_files = []
    for folder in subfolders:
        bookcorpus_files.extend(list(folder.glob('**/*.txt')))
    for file in bookcorpus_files:
        # read a random 512 words from the file
        with open(file, 'r') as f:
            text = f.read()
        sample = text[6000:16000]
        try:
            if detect(sample) != 'en':
                log_and_flush(f"Removing {file.name} from BooksCorpus because it is probably not English")
                bookcorpus_files.remove(file)
        except LangDetectException:
            log_and_flush(f"Removing {file.name} from BooksCorpus because langdetect failed")
            bookcorpus_files.remove(file)

    log_and_flush(f"Found {len(bookcorpus_files)} files in BooksCorpus")
    if test:
        bookcorpus_files = bookcorpus_files[:5]
    word_per_file = target_word_count // len(bookcorpus_files)
    log_and_flush(f"Aiming to sample {word_per_file} words from each of {len(bookcorpus_files)} files")
    # determine how many excerpts to extract
    num_excerpts = int(word_per_file // COUNT_PER_ROW) + 1
    log_and_flush(f"Sampling {num_excerpts} excerpts from each of {len(bookcorpus_files)} files")

    sampled_texts = []
    current_word_count = 0
    # caluculate the number of words to extract from each file
    for file in bookcorpus_files:
        log_and_flush(f"At file {file}")
        with open(file, 'r') as f:
            text = f.read()

        tokens = re.findall(r'\S+|\s+', text)
        log_and_flush(f"Words in file is {len(tokens) // 2}")

        # get num_excerpts random starting points that do not overlap in 512 token chunks
        possible_starts = list(range(0, len(tokens) - COUNT_PER_ROW * 2, COUNT_PER_ROW * 2))
        random.shuffle(possible_starts)

        for start in possible_starts[:num_excerpts]:
            text_entry = {
                'id': file.name + "_" + str(start),
                'text': ''.join(tokens[start:start + COUNT_PER_ROW * 2]),
                'word_count': COUNT_PER_ROW,
                'source': "books3"
            }
            sampled_texts.append(text_entry)
            current_word_count += COUNT_PER_ROW
        log_and_flush(f"Extracted {current_word_count} words")
    return sampled_texts, current_word_count


def create_balanced_dataset(total_word_count, test=False):
    if test:
        total_word_count = 512 * 10
    # Define the word ratio
    webtext_ratio = 1  # WebText to BooksCorpus ratio, wikipedia was 3.125

    # Calculate target word counts
    webtext_word_count = int(total_word_count * (webtext_ratio / (1 + webtext_ratio)))
    bookcorpus_word_count = total_word_count - webtext_word_count

    log_and_flush(f"Target word count for OpenWebText2: {webtext_word_count}")
    log_and_flush(f"Target word count for BooksCorpus: {bookcorpus_word_count}")

    random.seed(42)

    # Sample texts from each dataset
    sampled_web_texts, web_actual_word_count = sample_texts_from_webtext_dataset(webtext_word_count)
    sampled_bookcorpus_texts, bookcorpus_actual_word_count = (
        sample_texts_from_bookcorpus_dataset(bookcorpus_word_count, test=test))


    # Combine sampled texts into a single list
    combined_texts = sampled_web_texts + sampled_bookcorpus_texts

    # shuffle the list of dicts

    random.shuffle(combined_texts)

    # Print actual word counts achieved
    print(f"Actual word count from OpenWebText: {web_actual_word_count}")
    print(f"Actual word count from BooksCorpus: {bookcorpus_actual_word_count}")

    return combined_texts


def main(word_count=3_300_000_000, test=False):
    balanced_dataset = create_balanced_dataset(word_count, test=test)
    output_path = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/webbook"
    if test:
        output_path += "_test"
    save_to_huggingface_format(balanced_dataset, output_path)

    # Print out some information about the resulting dataset
    log_and_flush(f"Target word count was: {word_count}")
    log_and_flush(f"Number of examples: {len(balanced_dataset)}")
    log_and_flush(f"Sample entry: {balanced_dataset[0]}")  # Show the first dictionary entry

    return


if __name__ == '__main__':
    # create command line arguments: word count to sample
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the script in test mode")
    args = parser.parse_args()
    main(test=args.test)
