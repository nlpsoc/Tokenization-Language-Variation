"""
    used to create the sadiri classification dataset which is used to test the trained contrastive model
"""
import json
import re

from datasets import load_from_disk, DatasetDict, Dataset
import pandas as pd

from utility.env_variables import set_global_seed
from styletokenizer.whitespace_consts import APOSTROPHE_PATTERN
from styletokenizer.utility.datasets_helper import load_data

DEV_PATH = "../../data/UMich-AV/down_1/dev"
TRAIN_1_PATH = "../../data/UMich-AV/down_1/train"
TRAIN_10_PATH = "../../data/UMich-AV/down_10/train"
TRAIN_1_QUERY = "../../data/UMich-AV/down_1/train_queries.jsonl"
# original cluster location at /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1
TRAIN_1_CLUSTER = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1/train"
DEV_1_CLUSTER = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1/dev"

"""
    data has the form
     ['query_id', 'query_authorID', 'query_text', 'candidate_id', 'candidate_authorID', 'candidate_text']
"""


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


# Create pairs of texts
def _create_pairs(dataset, sources=None):
    # Set the seed once
    set_global_seed(42, False)

    df = pd.DataFrame(dataset)
    pairs = []
    queries = []
    candidates = []
    labels = []
    pair_sources = []
    for i, row in df.iterrows():
        # UMich dataset setup: query and candidate in the same row are a positive pair
        pairs.append((row['query_text'], row['candidate_text']))
        queries.append(row['query_text'])
        candidates.append(row['candidate_text'])
        labels.append(1)
        pair_sources.append(sources[i] if sources else None)
        # Add negative samples (pairs from different rows)
        #   get a random row
        neg_pair = False
        while not neg_pair:
            rand_row = df.sample(n=1)
            if rand_row['query_text'].values[0] != row['query_text']:
                pairs.append((row['query_text'], rand_row['query_text'].values[0]))
                queries.append(row['query_text'])
                candidates.append(rand_row['query_text'].values[0])
                labels.append(0)
                neg_pair = True
                pair_sources.append(sources[i] if sources else None)
    return (queries, candidates), labels, None if None in pair_sources else pair_sources


def create_singplesplit_sadiri_classification_dataset(train_path):
    (train_queries, train_candidates), train_labels, _ = _create_pairs(
        load_data(train_path))
    train_dataset = Dataset.from_dict({
        "query_text": train_queries,
        "candidate_text": train_candidates,
        "label": train_labels
    })
    return train_dataset
