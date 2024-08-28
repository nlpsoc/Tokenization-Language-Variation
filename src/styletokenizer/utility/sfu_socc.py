import pandas as pd

from styletokenizer.utility.sadiri import sample_texts_from_dataframe

SFUSOCC_PATH = "/shared/3/projects/hiatus/sfu-socc/SOCC/raw/gnm_articles.csv"
SFUSOCC_COMMENTS_PATH = "/shared/3/projects/hiatus/sfu-socc/SOCC/raw/gnm_comments.csv"
WORD_COUNT = 3_000_000


def sample_sfusocc_texts(word_count=WORD_COUNT, test=False):
    sfusocc_df = pd.read_csv(SFUSOCC_PATH)
    article_sample = sample_texts_from_dataframe(sfusocc_df, int(word_count // 2), "sfu-socc",
                                                 document_id='article_id', text_id='article_text', test=test)
    del sfusocc_df
    sfusocc_comments_df = pd.read_csv(SFUSOCC_COMMENTS_PATH)
    comment_sample = sample_texts_from_dataframe(sfusocc_comments_df, int(word_count // 2), "sfu-socc",
                                                 document_id='comment_id', text_id='comment_text', test=test)
    return article_sample + comment_sample
