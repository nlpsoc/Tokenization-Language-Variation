"""
    used to sample from nytimes
        see https://www.kaggle.com/datasets/benjaminawd/new-york-times-articles-comments-2020
"""
import styletokenizer.utility.AO3 as sadiri
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

nytimes_path = '/shared/3/projects/hiatus/data/raw_data/english/nytimes-articles-and-comments'
word_count = DOMAIN_WORDCOUNT_DICT["nytimes-articles-and-comments"]  # 25_000_000


def sample_nytimes_texts(test=False):
    return sadiri.sample_sadiri_texts([nytimes_path], [word_count], test=test)
