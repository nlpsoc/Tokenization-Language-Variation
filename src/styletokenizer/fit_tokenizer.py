import argparse
import bz2
import json
from tqdm import tqdm


import os

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
from transformers import AutoTokenizer
import datasets


def load_bz2_json_batch(file_path, batch_size=1000, total_lines=6459000):
    """
    Load a bz2 compressed JSON file in batches.

    :param file_path: Path to the bz2 compressed JSON file.
    :param batch_size: Number of lines to read in each batch.
    :param total_lines: Total number of lines to read from the file.
    :return: A generator yielding batches of JSON objects.
    """
    with bz2.open(file_path, 'rt') as f:
        batch = []
        for i, line in enumerate(f):
            if i >= total_lines:
                break
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def get_wiki_corpus_iterator(text_handle="text", test=False):
    """
    Get an iterator over the Wikipedia training corpus (https://huggingface.co/datasets/legacy-datasets/wikipedia)
        in batches of 1000, where each batch corresponds to 1 article
    :param text_handle:
    :param test:
    :return:
    """
    train_data = datasets.load_dataset('wikipedia', '20220301.en', split='train')
    # for i in tqdm(range(0, len(train_data), 1000), desc="Generating training corpus"):
    for i in range(0, len(train_data), 1000):
        yield train_data[i: i + 1000][text_handle]
        if test:
            break


def get_twitter_corpus_iterator(text_handle="text", test=False):
    """
        includes elements like
            {"created_at":"Wed Dec 01 04:53:30 +0000 2021","id":1465906883030568962,"id_str":"1465906883030568962","text":"RT @Divafwesh9: Drop handles &amp; Follow all who likes and Retweets this  \ud83c\udf29\ufe0f, \ud83c\udf43!","source":"\u003ca href=\"http:\/\/twitter.com\/download\/android\" rel=\"nofollow\"\u003eTwitter for Android\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":1454464894137470983,"id_str":"1454464894137470983","name":"\ud83e\udd8bSophia \u2764\ufe0f","screen_name":"Queen_Sophia1","location":null,"url":null,"description":"Live and love life | Actor | Model | DM Strictly for Business \ud83d\ude0b \ud83c\udf46\ud83c\udf51","translator_type":"none","protected":false,"verified":false,"followers_count":1151,"friends_count":1128,"listed_count":0,"favourites_count":4725,"statuses_count":3663,"created_at":"Sat Oct 30 15:07:29 +0000 2021","utc_offset":null,"time_zone":null,"geo_enabled":false,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"F5F8FA","profile_background_image_url":"","profile_background_image_url_https":"","profile_background_tile":false,"profile_link_color":"1DA1F2","profile_sidebar_border_color":"C0DEED","profile_sidebar_fill_color":"DDEEF6","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1458929350934507528\/dPx-bExU_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1458929350934507528\/dPx-bExU_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/1454464894137470983\/1636563851","default_profile":true,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null,"withheld_in_countries":[]},"geo":null,"coordinates":null,"place":null,"contributors":null,"retweeted_status":{"created_at":"Wed Dec 01 04:43:40 +0000 2021","id":1465904408621768704,"id_str":"1465904408621768704","text":"Drop handles &amp; Follow all who likes and Retweets this  \ud83c\udf29\ufe0f, \ud83c\udf43!","source":"\u003ca href=\"https:\/\/cheapbotsdonequick.com\" rel=\"nofollow\"\u003eCheap Bots, Done Quick!\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":1078021791086448640,"id_str":"1078021791086448640","name":"Diva\u265b","screen_name":"Divafwesh9","location":"Follow Back \ud83d\udd14","url":"http:\/\/www.diva.com","description":"Finding Love \u2764\u2764on Twitter doesn\u2019t mean Twitter is a Dating App. I am not here to find Love \ud83d\udc9e but Friends. #SmallAccounts \ud83d\ude03 \ud83d\ude03#Messi","translator_type":"none","protected":false,"verified":false,"followers_count":1158,"friends_count":1304,"listed_count":3,"favourites_count":1514,"statuses_count":15016,"created_at":"Wed Dec 26 20:16:38 +0000 2018","utc_offset":null,"time_zone":null,"geo_enabled":false,"lang":null,"contributors_enabled":false,"is_translator":false,"profile_background_color":"F5F8FA","profile_background_image_url":"","profile_background_image_url_https":"","profile_background_tile":false,"profile_link_color":"1DA1F2","profile_sidebar_border_color":"C0DEED","profile_sidebar_fill_color":"DDEEF6","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/1385246642689953795\/pv7Gumn3_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/1385246642689953795\/pv7Gumn3_normal.jpg","profile_banner_url":"https:\/\/pbs.twimg.com\/profile_banners\/1078021791086448640\/1619117226","default_profile":true,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null,"withheld_in_countries":[]},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":3,"favorite_count":2,"entities":{"hashtags":[],"urls":[],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"en"},"is_quote_status":false,"quote_count":0,"reply_count":0,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[],"urls":[],"user_mentions":[{"screen_name":"Divafwesh9","name":"Diva\u265b","id":1078021791086448640,"id_str":"1078021791086448640","indices":[3,14]}],"symbols":[]},"favorited":false,"retweeted":false,"filter_level":"low","lang":"en","timestamp_ms":"1638334410804"}
    :param text_handle:
    :param test:
    :return:
    """
    file_path = '/nfs/locker/twitter-decahose-locker/2021/decahose.2021-12-01.p2.bz2'
    # for batch in tqdm(load_bz2_json_batch(file_path, 1000), total=6459, desc="Loading Twitter data"):
    for batch in load_bz2_json_batch(file_path, 1000):
        for item in batch:
            yield item[text_handle]
        if test:
            break


def fit_wiki_tokenizer(corpus_iterator, vocab_size, dir_name, test=False):
    old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer = old_tokenizer.train_new_from_iterator(corpus_iterator, vocab_size=vocab_size,
                                                      length=(6459000 if not test else 1000))
    os.makedirs(dir_name, exist_ok=True)
    tokenizer.save_pretrained(f"{dir_name}")
    return tokenizer


def main(wiki=True, vocab_size=30000, test=False):
    dir_name = f"./llama3-tokenizer-{'wiki' if wiki else 'twitter'}-raw{'-test' if test else ''}/{vocab_size}"
    if wiki:
        training_corpus = get_wiki_corpus_iterator(test=test)
    else:
        training_corpus = get_twitter_corpus_iterator(test=test)

    tokenizer = fit_wiki_tokenizer(training_corpus, vocab_size, dir_name, test=test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--twitter", action="store_true")
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    main(wiki=not args.twitter, vocab_size=args.vocab_size, test=args.test)
