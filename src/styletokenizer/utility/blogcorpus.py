"""
    Sampling from blogcorpus,
        see https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
"""
import pandas as pd
from styletokenizer.utility.AO3 import sample_texts_from_dataframe
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

BLOG_PATH = "/shared/3/projects/hiatus/blog-corpus/blogtext.csv"
WORD_COUNT = DOMAIN_WORDCOUNT_DICT["blogcorpus"]  # 10_000_000


def sample_blogcorpus_texts(word_count=WORD_COUNT, test=False):
    # read in the blog corpus
    blog_df = pd.read_csv(BLOG_PATH)
    log_and_flush(f"Aiming to sample {word_count} words from blogcorpus")
    s_items, actual_word_count = sample_texts_from_dataframe(blog_df, word_count, "blogcorpus",
                                                             document_id='id', text_id='text', test=test)
    log_and_flush(f"Sampled {actual_word_count} words from blogcorpus")
    return s_items
