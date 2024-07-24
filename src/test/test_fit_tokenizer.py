from unittest import TestCase

from transformers import AutoTokenizer


class Test(TestCase):
    def test_fit_wiki_tokenizer(self):
        old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        tokenizer = old_tokenizer.train_new_from_iterator([["hello test 1", "hallo test 2"]], vocab_size=5)
