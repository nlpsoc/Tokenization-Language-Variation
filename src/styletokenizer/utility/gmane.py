import json
import random

from styletokenizer.utility.env_variables import make_text_fit_word_max

GMANE_PATH = "/shared/3/projects/hiatus/data/processed_data/english/gmane/gmane_cleaned_filtered.json"
WORD_COUNT = 150_000_000


def sample_from_json(word_count, json_path, id_column, text_column, source, test=False, w_langdetect=False):
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException

    if test:
        word_count = 100

    all_texts = []
    current_word_count = 0
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                cur_id = item.get(id_column, None)
                cur_text = item.get(text_column, None)

                if cur_id and cur_text:
                    all_texts.append({"id": cur_id, "text": cur_text})

    except Exception as e:
        print(f"Error processing file: {e}")
        return []

    random.shuffle(all_texts)

    sampled_items = []

    for item in all_texts:
        cur_text = item["text"]
        cur_text, cur_word_count = make_text_fit_word_max(cur_text)
        if cur_word_count > 1:
            if w_langdetect:
                try:
                    if not detect(cur_text) == 'en':
                        continue
                except LangDetectException:
                    continue

            # Add the sample to the list
            sampled_items.append({"id": item["id"], "text": cur_text, "word_count": cur_word_count,
                                  "source": source})
            current_word_count += cur_word_count

        if current_word_count >= word_count:
            break

    return sampled_items


def sample_gmane_texts(word_count=WORD_COUNT, test=False):
    return sample_from_json(word_count=word_count, json_path=GMANE_PATH,
                            id_column="time", text_column="text", source="gmane",
                            test=test)
