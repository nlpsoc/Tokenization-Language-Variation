from styletokenizer.utility.amazon import sample_from_gz_file_w_langdetect

GOOD_READ_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/raw/goodreads_reviews_dedup.json.gz"
WORD_COUNT = 50_000_000


def sample_goodreads_texts(word_count=WORD_COUNT, test=False):
    return sample_from_gz_file_w_langdetect(target_word_count=word_count, gz_path=GOOD_READ_PATH,
                                            id_column="review_id", text_column='review_text', source="goodreads",
                                            test=test)
