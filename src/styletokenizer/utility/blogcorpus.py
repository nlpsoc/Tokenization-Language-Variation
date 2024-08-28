import pandas as pd
from styletokenizer.utility.sadiri import sample_texts_from_dataframe

BLOG_PATH = "/shared/3/projects/hiatus/blog-corpus/blogtext.csv"
WORD_COUNT = 10_000_000


def sample_blogcorpus_texts(word_count=WORD_COUNT, test=False):
    # read in the blog corpus
    blog_df = pd.read_csv(BLOG_PATH)
    s_items, actual_word_count = sample_texts_from_dataframe(blog_df, word_count, "blogcorpus",
                                                             document_id='id', text_id='text', test=test)
    return s_items
