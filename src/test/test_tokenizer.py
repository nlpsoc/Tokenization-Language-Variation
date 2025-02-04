from unittest import TestCase

from huggingface_tokenizers import T5, HUGGINGFACE_TOKENIZERS, TRAINED_TOKENIZERS

import homoglyphs as hg
from fit_tokenizer import init_tokenizer_with_regex


class TestTokenizer(TestCase):

    # def setUp(self) -> None:
    #     from styletokenizer.tokenizer import TorchTokenizer
    #     self.tokenizers = []
    #     for tokenizer_name in HUGGINGFACE_TOKENIZERS:
    #         tokenizer = TorchTokenizer(tokenizer_name)
    #         self.tokenizers.append(tokenizer)
    #
    #     self.trained_tokenizers = []
    #     for trained_tokenizer_name in TRAINED_TOKENIZERS:
    #         tokenizer = TorchTokenizer(trained_tokenizer_name)
    #         self.trained_tokenizers.append(tokenizer)

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
        # unicode space is "\u0020", i.e., " " == "\u0020"
        test_str = ("well...\nI loves me a café \t i'm, i DON'T    or we’ve got 1000s\u00a0€ of emojis!!!\r‘🫨 😊 :) :D :((’   ")
        # test_str = (
        #     "Me ‘Time @ UMich's great!!😊 ’\nLearned 1000s thangs:\u00a0I'm I'M I’m café..  ")
        # test_str = "guy learnt\u00a0guitar\nThe man learned an instrument. "
        # test_str = ("😊,😭,🥺,🤣,❤️,✨,🙏,😍,🥰, Me ‘Learned 1000s thangs!!’\nI've learnt many things!")

        for tok_name, tokenizer in zip(TRAINED_TOKENIZERS, self.trained_tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            print(tokens)

        for tok_name, tokenizer in zip(HUGGINGFACE_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            token_ids = tokenizer.tokenizer.convert_tokens_to_ids(tokens)
            print(tokens)
            print(token_ids)

    def test_unicode(self):
        test_str = "Hello there,\u2003I &#108;ove emojis"
        # test_str = b"Hello there 😊"
        # test_str = test_str.encode('iso-8859-1')
        # test_str = test_str.encode('utf-8')
        # test_str = "� Hello there 😊".encode('utf-8').decode('ascii', errors='ignore')
        test_str = "Hello, world! &amp; Welcome to the &lt;coding&gt; world."
        print(test_str)
        for tok_name, tokenizer in zip(HUGGINGFACE_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.tokenize(test_str)
            token_ids = tokenizer.tokenizer.convert_tokens_to_ids(tokens)
            print(tokens)
            print(token_ids)

    def test_normalization(self):
        test_str = "Héllò hôw are ü?"
        print(test_str)
        for tok_name, tokenizer in zip(HUGGINGFACE_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            tokens = tokenizer.normalize(test_str)
            print(tokens)
            tokens = tokenizer.tokenize(tokens)
            print(tokens)

    def test_get_atoms(self):
        for tok_name, tokenizer in zip(HUGGINGFACE_TOKENIZERS, self.tokenizers):
            print(f"Tokenizer: {tok_name}")
            # Get the vocabulary
            vocab = tokenizer.get_vocab()
            print(f"Vocab size: {len(vocab)}")

            print("\U0001FAE8" in vocab)
            print("🫨" in vocab)
            print(f"UNK: {tokenizer.tokenize('🫨')}")
            print("\U0001F60A" in vocab)
            print("😊" in vocab)

            # Extract single characters from the vocabulary
            atoms = {token for token in vocab.keys() if len(token) == 1}
            latin = {token for token in vocab.keys() if len(token) == 1 and token.isalpha() and token.isascii()}
            print(tokenizer.tokenizer.all_special_tokens)
            print(tokenizer.tokenizer.special_tokens_map)

            print(len(atoms))
            print(atoms)

    def test_pre_tokenization(self):
        # test_str = ("\t😀 👍 ❤️ 😭 😁 😊 😂 well... \n\n I DON'T don't like café for $3000!! #lol 😊 😂 :) https://en.wikipedia.org/wiki/Sarcasm \r\r")
        test_str = (
            "well... $3000 for a tokenizer is cheapz #lol :)\n\nhttps://en.wikipedia.org/wiki/Sarcasm 😂")
        print(test_str)
        # pre-tokenize for ws, gpt2 and llama3
        for regex_pretok in ["no", "wsorg", "ws", "gpt2", "llama3"]:
            pretokenizer = init_tokenizer_with_regex(regex_pretok).pre_tokenizer
            print(pretokenizer.pre_tokenize_str(test_str))
        # from tokenizers.pre_tokenizers import Split
        # # create regex for \s
        # from tokenizers import Regex
        # ws_re = Regex(r'\s+')
        # isolated_pre_tokenizer = Split(ws_re, behavior='isolated')
        # print("isolated:", isolated_pre_tokenizer.pre_tokenize_str(test_str))
        # contiguous_pre_tokenizer = Split(ws_re, behavior='contiguous')
        # print("contiguous:", contiguous_pre_tokenizer.pre_tokenize_str(test_str))

    def test_homoglyphs(self):
        tokenizers = []
        for tokenizer_name in HUGGINGFACE_TOKENIZERS:
            tokenizer = TorchTokenizer(tokenizer_name)
            tokenizers.append(tokenizer)

        # testing different homoglyphs
        test_str = "Isle of dogs"
        homoglyphs = hg.Homoglyphs()
        test_str = ''.join(map(lambda c: (n := homoglyphs.get_combinations(c))[p if (p := n.index(c)+1) < len(n) else 0],
                               list(test_str)))
        print(test_str)

        for tok_name, tokenizer in zip(HUGGINGFACE_TOKENIZERS, tokenizers):
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