from datasets import load_from_disk
import pandas as pd

from utility.filesystem import set_global_seed

DEV_PATH = "../../data/UMich-AV/down_1/dev"
# original cluster location at /shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1

"""
    data has the form
     ['query_id', 'query_authorID', 'query_text', 'candidate_id', 'candidate_authorID', 'candidate_text']
"""


def load_1_dev_data():
    # loading follows same code as Kenan's
    # https://github.com/davidjurgens/sadiri/blob/main/src/style_content/poc/v1_no_adversarial/models.py#L23
    train_datatset = load_from_disk(DEV_PATH)['train']

    return train_datatset


# Create pairs of texts
def create_pairs(dataset):
    # Set the seed once
    set_global_seed(42, False)

    df = pd.DataFrame(dataset)
    pairs = []
    labels = []
    for i, row in df.iterrows():
        pairs.append((row['query_text'], row['candidate_text']))
        labels.append(1)
        # Add negative samples (pairs from different rows)
        #   get a random row
        neg_pair = False
        while not neg_pair:
            rand_row = df.sample(n=1)
            if rand_row['query_text'].values[0] != row['query_text']:
                pairs.append((row['query_text'], rand_row['query_text'].values[0]))
                labels.append(0)
                neg_pair = True
    return pairs, labels
