import json
import random

from styletokenizer.utility.env_variables import make_text_fit_word_max
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

GMANE_PATH = "/shared/3/projects/hiatus/data/processed_data/english/gmane/gmane_cleaned_filtered.json"
WORD_COUNT = DOMAIN_WORDCOUNT_DICT["gmane"]  # 150_000_000


def sample_from_json(target_word_count, json_path, id_column, text_column, source, test=False, w_langdetect=False):
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException

    if test:
        target_word_count = 100

    log_and_flush(f"Aiming to sample {target_word_count} words from {json_path}")
    sampled_items = []
    actual_word_count = 0

    with open(json_path, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
        random_indices = random.sample(range(total_lines), total_lines)  # Shuffle the line indices

        for idx in random_indices:
            if actual_word_count >= target_word_count:
                break

            line = json.loads(lines[idx])
            if not id_column:
                line_id = idx
            else:
                line_id = line.get(id_column)
            text = line.get(text_column)
            if not type(text) == str:
                continue
            text, cur_word_count = make_text_fit_word_max(text)
            if cur_word_count > 1:
                if w_langdetect:
                    try:
                        if not detect(text) == 'en':
                            continue
                    except LangDetectException:
                        continue

            sampled_items.append({
                "id": line_id,
                "text": text,
                "word_count": cur_word_count,
                "source": source,
                "domain": source
            })
            actual_word_count += cur_word_count

    log_and_flush(f"Sampled word count: {actual_word_count}")

    return sampled_items


def sample_gmane_texts(word_count=WORD_COUNT, test=False):
    return sample_from_json(target_word_count=word_count, json_path=GMANE_PATH,
                            id_column="time", text_column="text", source="gmane",
                            test=test)
