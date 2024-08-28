from styletokenizer.utility.amazon import sample_from_gz_file_w_langdetect

GOOD_READ_PATH = "/shared/3/datasets/goodreads/goodreads_reviews.json.gz"
WORD_COUNT = 55_000_000


def sample_goodreads_texts(word_count=WORD_COUNT, test=False):
    return sample_from_gz_file_w_langdetect(word_count=word_count, gz_path=GOOD_READ_PATH,
                                            id_column="review_id", text_column='review_text', source="goodreads",
                                            test=test)
