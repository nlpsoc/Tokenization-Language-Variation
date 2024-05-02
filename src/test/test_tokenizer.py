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
