import dask.dataframe as dd

WORD_COUNT = 100_000_000


def sample_YouTubeCommons_texts(required_word_count=WORD_COUNT):
    # Load the dataset using Dask
    dataset = dd.read_parquet('/shared/3/datasets/YouTube-Commons')
    print(f"YouTube-Commons loaded")

    # Filter the dataset where both transcription_language and original_language are 'en'
    filtered_dataset = dataset[(dataset['transcription_language'] == 'en') & (dataset['original_language'] == 'en')]
    print(f"YouTube-Commons filtered")

    # Debugging step: Check the number of rows after filtering
    filtered_count = filtered_dataset.shape[0].compute()
    print(f"Number of rows after filtering: {filtered_count}")

    if filtered_count == 0:
        raise ValueError("No data found after filtering. Check the filter conditions or data content.")

    # Shuffle the dataset
    shuffled_dataset = filtered_dataset.sample(frac=1, random_state=42)

    # Compute the shuffled dataset
    filtered_df = shuffled_dataset.compute()

    # Initialize variables to store the sampled texts and the accumulated word count
    total_word_count = 0
    sampled_items = []
    # sampled_texts = []
    # sampled_ids = []
    # sampled_word_count = []

    # Iterate through the shuffled dataframe and sample texts until the required word count is reached
    for _, row in filtered_df.iterrows():
        sampled_items.append({
            "text": row['text'], "word_count": row['word_count'], "id": row['video_id'], "source": "YouTubeCommons",
            "domain": "YouTubeCommons"
        })
        # sampled_texts.append(row['text'])
        # sampled_ids.append(row['video_id'])
        # sampled_word_count.append(row['word_count'])
        total_word_count += row['word_count']
        if total_word_count >= required_word_count:
            break

    # Ensure that we have sampled enough texts to meet the word count requirement
    if total_word_count < required_word_count:
        raise ValueError("Not enough data to meet the required word count")

    print(f"Sampled word count: {total_word_count}")

    return sampled_items
    # {
    #     "id": sampled_ids,
    #     "domain": ["YouTubeCommons"] * len(sampled_texts),
    #     "source": ["YouTubeCommons"] * len(sampled_texts),
    #     "word_count": sampled_word_count,
    #     "text": sampled_texts,
    # }
