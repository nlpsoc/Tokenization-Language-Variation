from unittest import TestCase

from huggingface_tokenizers import T5, ALL_TOKENIZERS
from styletokenizer.tokenizer import TorchTokenizer
import homoglyphs as hg


class TestTokenizer(TestCase):

    def setUp(self) -> None:
        self.tokenizers = []
        for tokenizer_name in ALL_TOKENIZERS:
            tokenizer = TorchTokenizer(tokenizer_name)
            self.tokenizers.append(tokenizer)

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
        test_str = ("well...\u2003I aready can`t say i don'T or can't love, 100s € of emojis!!! 😊 :) :D :((")
        print(test_str)

        for tok_name, tokenizer in zip(ALL_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            print(tokens)

    def test_unicode(self):
        test_str = "Hello there,\u2003I &#108;ove emojis"
        # test_str = b"Hello there 😊"
        # test_str = test_str.encode('iso-8859-1')
        # test_str = test_str.encode('utf-8')
        # test_str = "� Hello there 😊".encode('utf-8').decode('ascii', errors='ignore')
        test_str = "Hello, world! &amp; Welcome to the &lt;coding&gt; world."
        print(test_str)
        for tok_name, tokenizer in zip(ALL_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            print(tokens)

    def test_normalization(self):
        test_str = "Héllò hôw are ü?"
        print(test_str)
        for tok_name, tokenizer in zip(ALL_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.normalize(test_str)
            print(tokens)
            tokens = tokenizer.tokenize(tokens)
            print(tokens)

    def test_get_atoms(self):
        for tok_name, tokenizer in zip(ALL_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            # Get the vocabulary
            vocab = tokenizer.get_vocab()

            # Extract single characters from the vocabulary
            atoms = {token for token in vocab.keys() if len(token) == 1}
            latin = {token for token in vocab.keys() if len(token) == 1 and token.isalpha() and token.isascii()}

            print(len(atoms))
            print(latin)

    def test_homoglyphs(self):
        tokenizers = []
        for tokenizer_name in ALL_TOKENIZERS:
            tokenizer = TorchTokenizer(tokenizer_name)
            tokenizers.append(tokenizer)

        # testing different homoglyphs
        test_str = "Isle of dogs"
        homoglyphs = hg.Homoglyphs()
        test_str = ''.join(map(lambda c: (n := homoglyphs.get_combinations(c))[p if (p := n.index(c)+1) < len(n) else 0],
                               list(test_str)))
        print(test_str)

        for tok_name, tokenizer in zip(ALL_TOKENIZERS, tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            print(tokens)




        # tokens = rob_tokenizer.tokenize("toot-toot")
        # print(tokens)
        # tokens = rob_tokenizer.tokenize("punktuation")
        # print(tokens)
        # tokens = rob_tokenizer.tokenize("Good job - well done!")
        # print(tokens)
        # tokens = rob_tokenizer.tokenize("'tis'")
        # print(tokens)
        #
        # xlmroberta_tokenizer = TorchTokenizer("xlm-roberta-base")
        # tokens = xlmroberta_tokenizer.tokenize("outtake")
        # print(tokens)
        #
        # llama38b_tokenizer = TorchTokenizer("meta-llama/Meta-Llama-3-8B")
        # tokens = llama38b_tokenizer.tokenize("outtake")
        # print(tokens)