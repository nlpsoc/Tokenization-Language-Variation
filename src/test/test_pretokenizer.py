from unittest import TestCase

import homoglyphs as hg
from fit_tokenizer import init_tokenizer_with_regex


class TestTokenizer(TestCase):

    def test_pre_tokenization(self):
        test_str = (
            "well... $3000 for a tokenizer is cheapz #lol :)\n\nhttps://en.wikipedia.org/wiki/Sarcasm 😂")
        test_str = "couldn’t"
        print(test_str)
        # pre-tokenize for ws, gpt2 and llama3
        for regex_pretok in ["no", "wsorg", "ws", "gpt2", "llama3"]:
            pretokenizer = init_tokenizer_with_regex(regex_pretok).pre_tokenizer
            print(pretokenizer.pre_tokenize_str(test_str))

