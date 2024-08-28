AMAZON_PATH = "/shared/3/datasets/amazon-reviews/All_Amazon_Review.json.gz"
WORD_COUNT = 35_000_000


def sample_amazon_texts(test=False):
    import pandas as pd
    import gzip
    import langdetect

    amazon_df = pd.read_json(AMAZON_PATH, lines=True)
    return sadiri.sample_texts_from_dataframe(amazon_df, WORD_COUNT, "Amazon", test=test)
