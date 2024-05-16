from unittest import TestCase
from styletokenizer.tokenizer import TorchTokenizer


class TestTokenizer(TestCase):

    def test_tokenizer(self):
        tokenizer = TorchTokenizer("bert-base-uncased")
        tokens = tokenizer.tokenize("Hello, my dog is cute")
        print(tokens)

        self.assertEqual(tokens, ['hello', ',', 'my', 'dog', 'is', 'cute'])

        # test on a list
        tokens = tokenizer.tokenize(["Hello, my dog is cute", "I like to eat ice cream"])
        print(tokens)
        self.assertEqual(tokens, [['hello', ',', 'my', 'dog', 'is', 'cute'], ['i', 'like', 'to', 'eat', 'ice', 'cream']])

    def test_robrta_tokenizer(self):
        test_str = "well... I aready can`t say i don'T or can't love, emojis!!! 😊 :) :D :(("
        print(test_str)

        rob_tokenizer = TorchTokenizer("roberta-base")
        tokens = rob_tokenizer.tokenize(test_str)
        print(tokens)

        bert_tokenizer = TorchTokenizer("bert-base-cased")
        tokens = bert_tokenizer.tokenize(test_str)
        print(tokens)

        lama38b_tokenizer = TorchTokenizer("meta-llama/Meta-Llama-3-8B")
        tokens = lama38b_tokenizer.tokenize(test_str)
        print(tokens)


        tokens = rob_tokenizer.tokenize("toot-toot")
        print(tokens)
        tokens = rob_tokenizer.tokenize("punktuation")
        print(tokens)
        tokens = rob_tokenizer.tokenize("Good job - well done!")
        print(tokens)
        tokens = rob_tokenizer.tokenize("'tis'")
        print(tokens)

        xlmroberta_tokenizer = TorchTokenizer("xlm-roberta-base")
        tokens = xlmroberta_tokenizer.tokenize("outtake")
        print(tokens)

        llama38b_tokenizer = TorchTokenizer("meta-llama/Meta-Llama-3-8B")
        tokens = llama38b_tokenizer.tokenize("outtake")
        print(tokens)