import re

from datasets import load_from_disk
import pandas as pd

from utility.filesystem import set_global_seed
from whitespace_consts import APOSTROPHE_PATTERN

DEV_PATH = "../../data/UMich-AV/down_1/dev"
TRAIN_1_PATH = "../../data/UMich-AV/down_1/train"
TRAIN_10_PATH = "../../data/UMich-AV/down_10/train"
# original cluster location at /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1
TRAIN_1_CLUSTER = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1/train"
DEV_1_CLUSTER = "/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1/dev"

"""
    data has the form
     ['query_id', 'query_authorID', 'query_text', 'candidate_id', 'candidate_authorID', 'candidate_text']
"""


def load_1_dev_data():
    # loading follows same code as Kenan's
    # https://github.com/davidjurgens/sadiri/blob/main/src/style_content/poc/v1_no_adversarial/models.py#L23
    from styletokenizer.utility.filesystem import on_cluster
    if not on_cluster():
        train_datatset = load_from_disk(DEV_PATH)['train']
    else:
        train_datatset = load_from_disk(DEV_1_CLUSTER)['train']
    return train_datatset


def load_1_train_data():
    from styletokenizer.utility.filesystem import on_cluster
    if not on_cluster():
        train_dataset = load_from_disk(TRAIN_1_PATH)['train']
    else:
        train_dataset = load_from_disk(TRAIN_1_CLUSTER)['train']
    return train_dataset

def load_10_train_data():
    train_dataset = load_from_disk(TRAIN_10_PATH)['train']
    return train_dataset


# Create pairs of texts
def _create_pairs(dataset):
    # Set the seed once
    set_global_seed(42, False)

    df = pd.DataFrame(dataset)
    pairs = []
    queries = []
    candidates = []
    labels = []
    for i, row in df.iterrows():
        # UMich dataset setup: query and candidate in the same row are a positive pair
        pairs.append((row['query_text'], row['candidate_text']))
        queries.append(row['query_text'])
        candidates.append(row['candidate_text'])
        labels.append(1)
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
    return (queries, candidates), labels


def get_1_dev_pairs():
    dataset = load_1_dev_data()
    return _create_pairs(dataset)


def get_1_dev_dataframe():
    pairs, labels = get_1_dev_pairs()
    return pd.DataFrame({"query": pairs[0], "candidate": pairs[1], "label": labels})


def get_1_train_pairs():
    dataset = load_1_train_data()
    return _create_pairs(dataset)

def get_10_train_pairs():
    dataset = load_10_train_data()
    return _create_pairs(dataset)


def get_1_train_dataframe():
    pairs, labels = get_1_train_pairs()
    return pd.DataFrame({"query": pairs[0], "candidate": pairs[1], "label": labels})

def get_10_train_dataframe():
    pairs, labels = get_10_train_pairs()
    return pd.DataFrame({"query": pairs[0], "candidate": pairs[1], "label": labels})


def find_av_matches(df, apostrophe_pattern=APOSTROPHE_PATTERN):
    # Function to find and extract context around apostrophes in a column
    def extract_apostrophe_context_with_unicode(text, pattern, context=5):
        matches = []
        for match in re.finditer(pattern, text):
            start = max(0, match.start() - context)
            end = min(len(text), match.end() + context)
            context_str = text[start:match.start()] + match.group() + " (U+" + format(ord(match.group()),
                                                                                      '04X') + ")" + text[
                                                                                                     match.end():end]
            matches.append((match.group(), context_str))
        return matches

    def find_apostrophes(df, column_name, pattern):
        df['apostrophe_context'] = df[column_name].apply(lambda x: extract_apostrophe_context_with_unicode(x, pattern))
        return df

    result_df = find_apostrophes(df, 'query', apostrophe_pattern)
    result_df = result_df[result_df['apostrophe_context'].apply(bool)]
    # Explode the context column to separate rows for each match
    exploded_df = result_df.explode('apostrophe_context')
    # Extract the apostrophe and context separately
    exploded_df['apostrophe'] = exploded_df['apostrophe_context'].apply(lambda x: x[0])
    exploded_df['context'] = exploded_df['apostrophe_context'].apply(lambda x: x[1])

    # shuffle the dataframe
    exploded_df = exploded_df.sample(frac=1).reset_index(drop=True)

    # Group by apostrophe type and collect examples
    grouped = exploded_df.groupby('apostrophe')['context'].apply(list).reset_index()

    # Function to print number of examples and up to 10 examples per apostrophe type
    def print_examples_per_apostrophe_type(grouped_df, max_examples=10):
        for index, row in grouped_df.iterrows():
            apostrophe = row['apostrophe']
            examples = row['context']
            num_examples = len(examples)
            print(f"Unicode: {apostrophe} (U+{ord(apostrophe):04X}) - {num_examples} examples")
            for example in examples[:max_examples]:
                print(f"  Example: {example}")
            print()

    # Display the examples
    print_examples_per_apostrophe_type(grouped)
