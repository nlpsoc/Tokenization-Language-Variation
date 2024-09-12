from unittest import TestCase
from src.eval_tokenizer import calc_renyi_efficiency, calc_seq_len, calc_avg_tok_per_word, main, calc_precentile_freq
import tqdm

class Test(TestCase):
    def test_calc_renyi_efficiency(self):
        print(calc_seq_len("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
                                   "../../data/fitting-corpora/mixed"))
        print(calc_seq_len("../../data/tokenizer/twitter-gpt2-32000/tokenizer.json",
                                   "../../data/fitting-corpora/mixed"))
        print(calc_seq_len("../../data/tokenizer/wikipedia-gpt2-32000/tokenizer.json",
                                   "../../data/fitting-corpora/mixed"))
        print(calc_seq_len("../../data/tokenizer/mixed-ws-32000/tokenizer.json",
                                   "../../data/fitting-corpora/mixed"))
        print(calc_seq_len("../../data/tokenizer/mixed-llama3-32000/tokenizer.json",
                                   "../../data/fitting-corpora/mixed"))
        # print(calc_renyi_efficiency("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                             "../../data/fitting-corpora/mixed"))
        # print(calc_seq_len("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                    "../../data/fitting-corpora/mixed"))
        # print(calc_avg_tok_per_word("../../data/tokenizer/mixed-gpt2-32000/tokenizer.json",
        #                             "../../data/fitting-corpora/mixed"))

    def test_eval_tokenizer(self):
        tqdm.tqdm = lambda *args, **kwargs: iter(args[0])
        main()
