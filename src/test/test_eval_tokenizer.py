from unittest import TestCase
from eval_tokenizer import calc_seq_len_from_path, main
import tqdm
from utility.webbook import CORPORA_WEBBOOK


class Test(TestCase):
    def test_calc_renyi_efficiency(self):
        twitter_path = "../../data/fitting-corpora/twitter"
        wikipedia_path = "../../data/fitting-corpora/wikipedia"
        mixed_path = "../../data/fitting-corpora/mixed"
        webbook_path = CORPORA_WEBBOOK
        for data_path in [twitter_path, wikipedia_path, mixed_path, webbook_path]:
            print(f"\n{data_path}")
            print(calc_seq_len_from_path("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
                               "../../data/fitting-corpora/mixed"))

    def test_eval_tokenizer(self):
        tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
        main()
