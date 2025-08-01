"""
    used to sample from realnews
        https://github.com/rowanz/grover/tree/master/realnews
"""
from styletokenizer.utility.gmane import sample_from_json
from styletokenizer.utility.mixed import DOMAIN_WORDCOUNT_DICT

REALNEWS_PATH = "/shared/3/datasets/realnews/realnews/realnews.jsonl"
WORD_COUNT = DOMAIN_WORDCOUNT_DICT["realnews"]  #  150_000_000


def sample_realnews_texts(word_count=WORD_COUNT, test=False):
    return sample_from_json(target_word_count=word_count, json_path=REALNEWS_PATH,
                            id_column="warc_date", text_column="text", source="realnews",
                            test=test)
