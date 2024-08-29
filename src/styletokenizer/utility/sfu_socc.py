import pandas as pd

from styletokenizer.utility.sadiri import sample_texts_from_dataframe
from styletokenizer.utility.custom_logger import log_and_flush

SFUSOCC_PATH = "/shared/3/projects/hiatus/sfu-socc/SOCC/raw/gnm_articles.csv"
SFUSOCC_COMMENTS_PATH = "/shared/3/projects/hiatus/sfu-socc/SOCC/raw/gnm_comments.csv"
WORD_COUNT = 3_000_000


def sample_sfusocc_texts(word_count=WORD_COUNT, test=False):
    sfusocc_df = pd.read_csv(SFUSOCC_PATH)
    log_and_flush(f"Aiming to sample {int(word_count // 2)} words from sfu-socc")
    article_sample, article_word_count = sample_texts_from_dataframe(sfusocc_df, int(word_count // 2), "sfu-socc",
                                                 document_id='article_id', text_id='article_text', test=test)
    log_and_flush(f"Sampled {article_word_count} words from sfu-socc articles")
    del sfusocc_df
    sfusocc_comments_df = pd.read_csv(SFUSOCC_COMMENTS_PATH, low_memory=False)
    log_and_flush(f"Aiming to sample {int(word_count // 2)} words from sfu-socc comments")
    comment_sample, comment_word_count = sample_texts_from_dataframe(sfusocc_comments_df, int(word_count // 2), "sfu-socc",
                                                 document_id='comment_id', text_id='comment_text', test=test)
    log_and_flush(f"Sampled {comment_word_count} words from sfu-socc comments")
    return article_sample + comment_sample
