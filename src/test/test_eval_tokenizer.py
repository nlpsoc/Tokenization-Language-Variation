from unittest import TestCase
from src.eval_tokenizer import calc_renyi_efficiency_from_path, calc_seq_len, calc_avg_tok_per_word_from_path, main, calc_precentile_freq
import tqdm
from styletokenizer.utility.preptraining_corpora import CORPORA_WEBBOOK


class Test(TestCase):
    def test_calc_renyi_efficiency(self):
        # print(calc_seq_len("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                            "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/twitter-gpt2-32000/tokenizer.json",
        #                            "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/wikipedia-gpt2-32000/tokenizer.json",
        #                            "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/mixed-ws-32000/tokenizer.json",
        #                            "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/mixed-llama3-32000/tokenizer.json",
        #                            "../../data/fitting-corpora/mixed"))
        tokenizer_path = "../../data/tokenizer/mixed-gpt2-64000/tokenizer.json"
        twitter_path = "../../data/fitting-corpora/twitter"
        wikipedia_path = "../../data/fitting-corpora/wikipedia"
        mixed_path = "../../data/fitting-corpora/mixed"
        webbook_path = CORPORA_WEBBOOK
        for data_path in [twitter_path, wikipedia_path, mixed_path, webbook_path]:
            print(f"\n{data_path}")
            print(calc_renyi_efficiency_from_path(tokenizer_path, data_path))
            print(calc_avg_tok_per_word_from_path(tokenizer_path, data_path))
        # print(calc_avg_tok_per_word("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                             "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                    "../../data/fitting-corpora/mixed"))

    def test_eval_tokenizer(self):
        tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
        main()
