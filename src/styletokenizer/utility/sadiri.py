import os
import pandas as pd
from collections import defaultdict
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.env_variables import make_text_fit_word_max

project_base = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/"
SET_PATHS = [  # "reddit",
    "ao3",
    # "realnews", "nytimes-articles-and-comments", "sfu-socc", "goodreads",
    # "amazon", "gmane", "blogcorpus"
]  # "bookcorpus"
SET_PATHS = [project_base + folder_name for folder_name in SET_PATHS]
WORD_COUNTS = [  # 249000000,
    150_000_000,  # increase by 50M because bookcorpus was removed
    # 169000000,
    # 24131163,
    # 3007117,
    # 53683977,
    # 31650279,
    # 141837101,
    # 8189607
]  # 50000000


def sample_texts_from_dataframe(data_df, target_word_count, dataset_name,
                                document_id='documentID', text_id='fullText', test=False):
    sampeled_items = []
    sampled_texts = []
    document_ids = []
    sampled_word_counts = []
    current_word_count = 0
    data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffle the texts
    for idx, row in data_df.iterrows():
        text = row[text_id]
        text, word_count = make_text_fit_word_max(text)
        sampeled_items.append({
            "text": text, "word_count": word_count, "id": row[document_id], "source": "SADIRI",
            "domain": dataset_name})
        sampled_texts.append(text)
        document_ids.append(row[document_id])
        sampled_word_counts.append(word_count)
        current_word_count += word_count
        if (current_word_count >= target_word_count) or test:
            break
    return sampeled_items, current_word_count


def sample_sadiri_texts(dataset_paths=SET_PATHS, word_samples=WORD_COUNTS, test=False,
                        extract_from_jsonl=['train_queries.jsonl', 'train_candidates.jsonl']):
    sampled_items = []

    for dataset_path, word_count in zip(dataset_paths, word_samples):
        dataset_name = os.path.basename(dataset_path)

        combined_df = pd.DataFrame()

        log_and_flush(f"Loading data from {dataset_path}")
        # Combine both JSONL files into one dataframe

        for file_name in extract_from_jsonl:  # 'corpus.jsonl' does not exist for all datasets
            file_path = os.path.join(dataset_path, file_name)
            if os.path.exists(file_path):
                data_df = pd.read_json(file_path, lines=True)
                log_and_flush(f"Loaded {len(data_df)} rows from {file_path}")
                combined_df = pd.concat([combined_df, data_df], ignore_index=True)
            else:
                log_and_flush(f"{file_name} does not exist in {dataset_path}.")

        current_sample, current_word_count = sample_texts_from_dataframe(combined_df, word_count, dataset_name,
                                                                         test=test)
        log_and_flush(f"Sampled {len(current_sample)} words from {dataset_name}")
        sampled_items += current_sample

    return sampled_items
