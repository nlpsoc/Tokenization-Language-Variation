import json
import pandas as pd

DEV_PATH = "../../data/UMich-AA/long-reddit/eval.jsonl"
COMBINED_PATH = "../../data/UMich-AA/long-reddit/combined.jsonl"


def load_dev_data():
    with open(DEV_PATH, 'r') as f:
        data = [json.loads(line) for line in f]

    # create pandas dataframe
    df = pd.DataFrame(data)

    return df
