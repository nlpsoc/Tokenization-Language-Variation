"""
    used to sample from amazon reviews
        see also: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
"""
import gzip
import json
import random

from utility.datasets_helper import make_text_fit_word_max
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

AMAZON_PATH = "/shared/3/datasets/amazon-reviews/All_Amazon_Review.json.gz"
WORD_COUNT = DOMAIN_WORDCOUNT_DICT["amazon"]  # 50_000_000


def sample_amazon_texts(word_count=WORD_COUNT, test=False):
    id_column = "unixReviewTime"
    text_column = "reviewText"
    gz_path = AMAZON_PATH
    source = "amazon"

    return sample_from_gz_file_w_langdetect(word_count, gz_path, id_column, text_column, source, test=test)


def sample_from_gz_file_w_langdetect(target_word_count, gz_path, id_column, text_column, source, test=False):
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException

    if test:
        target_word_count = 100

    log_and_flush(f"Loading data from {gz_path}")
    all_reviews = []
    actual_word_count = 0
    try:
        # Load entire file into memory
        with gzip.open(gz_path, 'rt') as f:
            for line in f:
                try:
                    # Parse each line as JSON
                    review = json.loads(line)
                    cur_id = review.get(id_column, None)
                    cur_text = review.get(text_column, None)

                    # Check if both id and text are available
                    if cur_id and cur_text:
                        all_reviews.append({"id": cur_id, "text": cur_text})

                except json.JSONDecodeError:
                    # Skip any lines that are not valid JSON
                    continue

    except Exception as e:
        print(f"Error processing file: {e}")
        return []

    # Shuffle the entire dataset
    random.shuffle(all_reviews)

    log_and_flush(f"Aiming to sample {target_word_count} words from {gz_path}")
    sampled_items = []

    # Process the shuffled data and sample based on word count
    for review in all_reviews:
        cur_text = review["text"]

        try:
            # Detect language and filter for English
            if detect(cur_text) == 'en':
                cur_text, cur_word_count = make_text_fit_word_max(cur_text)

                # Add the sample to the list
                sampled_items.append({"id": review["id"], "text": cur_text,
                                      "word_count": cur_word_count,
                                      "source": source, "domain": source})

                # Accumulate the word count
                actual_word_count += cur_word_count

                # Stop if the target word count is reached
                if actual_word_count >= target_word_count:
                    break
        except LangDetectException:
            # Skip if language detection fails
            continue
    log_and_flush(f"Sampled word count: {actual_word_count}")

    return sampled_items
