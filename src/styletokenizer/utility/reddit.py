from styletokenizer.utility.sadiri import sample_sadiri_texts

REDDIT_DIR_PATH = "/shared/3/projects/hiatus/data/raw_data/english/reddit"
WORD_COUNT = 250_000_000


def sample_reddit_texts(word_count=WORD_COUNT, test=False):
    return sample_sadiri_texts(word_samples=[word_count], dataset_paths=[REDDIT_DIR_PATH],
                               extract_from_jsonl=["corpus_original.jsonl"], test=test)
