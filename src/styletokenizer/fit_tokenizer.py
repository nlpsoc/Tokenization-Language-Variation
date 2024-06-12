import argparse
import bz2
import json
from tqdm import tqdm


import os

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
from transformers import AutoTokenizer
import datasets


def load_bz2_json_batch(file_path, batch_size=1000, total_lines=6459000):
    """
    Load a bz2 compressed JSON file in batches.

    :param file_path: Path to the bz2 compressed JSON file.
    :param batch_size: Number of lines to read in each batch.
    :param total_lines: Total number of lines to read from the file.
    :return: A generator yielding batches of JSON objects.
    """
    with bz2.open(file_path, 'rt') as f:
        batch = []
        for i, line in enumerate(f):
            if i >= total_lines:
                break
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def get_wiki_corpus_iterator(text_handle="text", test=False):
    train_data = datasets.load_dataset('wikipedia', '20220301.en', split='train')
    for i in tqdm(range(0, len(train_data), 1000), desc="Generating training corpus"):
        yield train_data[i: i + 1000][text_handle]
        if test:
            break


def get_twitter_corpus_iterator(text_handle="text", test=False):
    file_path = '/nfs/locker/twitter-decahose-locker/2021/decahose.2021-12-01.p2.bz2'
    for batch in tqdm(load_bz2_json_batch(file_path, 1000), total=6459, desc="Loading Twitter data"):
        for item in batch:
            yield item[text_handle]
        if test:
            break


def fit_wiki_tokenizer(corpus_iterator, vocab_size, dir_name, test=False):
    old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = old_tokenizer.train_new_from_iterator(corpus_iterator, vocab_size=vocab_size,
                                                      ength=(6459000 if not test else 1000))
    os.makedirs(dir_name, exist_ok=True)
    tokenizer.save_pretrained(f"{dir_name}")
    return tokenizer


def main(wiki=True, vocab_size=30000, test=False):
    dir_name = f"./llama3-tokenizer-{'wiki' if wiki else 'twitter'}-raw{'-test' if test else ''}/{vocab_size}"
    if wiki:
        training_corpus = get_wiki_corpus_iterator(test=test)
    else:
        training_corpus = get_twitter_corpus_iterator(test=test)

    tokenizer = fit_wiki_tokenizer(training_corpus, vocab_size, dir_name, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(wiki=not args.twitter, vocab_size=args.vocab_size, test=args.test)
