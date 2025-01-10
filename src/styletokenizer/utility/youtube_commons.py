import dask.dataframe as dd
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.env_variables import make_text_fit_word_max
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

WORD_COUNT = DOMAIN_WORDCOUNT_DICT["YouTubeCommons"]  # 100_000_000


def sample_YouTubeCommons_texts(required_word_count=WORD_COUNT, test=False):
    # Load the dataset using Dask
    dataset = dd.read_parquet('/shared/3/datasets/YouTube-Commons')
    log_and_flush(f"YouTube-Commons loaded")

    # Filter the dataset where both transcription_language and original_language are 'en'
    filtered_dataset = dataset[(dataset['transcription_language'] == 'en') & (dataset['original_language'] == 'en')]
    log_and_flush(f"YouTube-Commons filtered")

    # Debugging step: Check the number of rows after filtering
    filtered_count = filtered_dataset.shape[0].compute()
    log_and_flush(f"Number of rows after filtering: {filtered_count}")

    if filtered_count == 0:
        raise ValueError("No data found after filtering. Check the filter conditions or data content.")

    # Shuffle the dataset
    shuffled_dataset = filtered_dataset.sample(frac=1, random_state=42)

    # Compute the shuffled dataset
    filtered_df = shuffled_dataset.compute()

    log_and_flush(f"Target word count: {required_word_count}")
    # Initialize variables to store the sampled texts and the accumulated word count
    total_word_count = 0
    sampled_items = []

    # Iterate through the shuffled dataframe and sample texts until the required word count is reached
    for _, row in filtered_df.iterrows():
        text, text_word_count = make_text_fit_word_max(row['text'])
        sampled_items.append({
            "text": text, "word_count": text_word_count, "id": row['video_id'], "source": "YouTubeCommons",
            "domain": "YouTubeCommons"
        })
        total_word_count += text_word_count
        if (total_word_count >= required_word_count) or test:
            break

    # Ensure that we have sampled enough texts to meet the word count requirement
    if (not test) and (total_word_count < required_word_count):
        log_and_flush("WARNING: Not enough data to meet the required word count")

    log_and_flush(f"Sampled word count: {total_word_count}")

    return sampled_items
