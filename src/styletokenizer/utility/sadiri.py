import os
import pandas as pd
from collections import defaultdict

project_base = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/"
SET_PATHS = ["reddit", "ao3", "bookcorpus", "realnews", "nytimes-articles-and-comments", "sfu-socc", "goodreads",
                "amazon", "gmane", "blogcorpus"]
SET_PATHS = [project_base + folder_name for folder_name in SET_PATHS]
WORD_COUNTS = [249000000,
               100000000,
               50000000,
               169000000,
               24131163,
               3007117,
               53683977,
               31650279,
               141837101,
               8189607]


def sample_texts_from_dataframe(data_df, target_word_count):
    sampled_texts = []
    document_ids = []
    sampled_word_counts = []
    current_word_count = 0
    data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffle the texts
    for idx, row in data_df.iterrows():
        text = row['fullText']
        words = text.split()
        word_count = len(words)
        sampled_texts.append(text)
        document_ids.append(row['documentID'])
        sampled_word_counts.append(word_count)
        current_word_count += word_count
        if current_word_count >= target_word_count:
            break
    return {
        "sampled_texts": sampled_texts,
        "sampled_word_count": sampled_word_counts,
        "document_ids": document_ids
    }


def sample_sadiri_texts(dataset_paths=SET_PATHS, word_samples=WORD_COUNTS):
    data_samples = defaultdict(list)
    total_word_count = defaultdict(int)

    for dataset_path, word_count in zip(dataset_paths, word_samples):
        dataset_name = os.path.basename(dataset_path)

        combined_df = pd.DataFrame()

        # Combine both JSONL files into one dataframe
        for file_name in ['train_queries.jsonl', 'train_candidates.jsonl']:
            file_path = os.path.join(dataset_path, file_name)
            if os.path.exists(file_path):
                data_df = pd.read_json(file_path, lines=True)
                print(f"Loaded {len(data_df)} rows from {file_path}")
                combined_df = pd.concat([combined_df, data_df], ignore_index=True)
            else:
                print(f"{file_name} does not exist in {dataset_path}.")

        sample_dict = sample_texts_from_dataframe(combined_df, word_count)

        data_samples['source'] += ["SADIRI" for _ in range(len(sample_dict))]
        data_samples['domain'] += [dataset_name for _ in range(len(sample_dict))]
        data_samples['text'] += sample_dict["sampled_texts"]
        data_samples['id'] += sample_dict["document_ids"]
        data_samples['word_count'] += sample_dict["sampled_word_count"]
        total_word_count[dataset_name] += sample_dict["sampled_word_count"]
        print(f"Total words collected for {dataset_name}: {total_word_count[dataset_name]}")

    print(data_samples)
