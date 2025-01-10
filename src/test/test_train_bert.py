import subprocess
from unittest import TestCase

from train_bert import main


class Test(TestCase):
    def test_main(self):
        tokenizer = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/tokenizer/wikipedia-gpt2-32000"
        word_count = 750_000_000
        steps = 45_000
        random_seed = 42
        output_base_folder = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/src/test/output"
        data_path = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/data/fitting-corpora/mixed"
        batch_size = 32
        model_size = 110
        test = True

        main(tokenizer_path=tokenizer, word_count=word_count, steps=steps, random_seed=random_seed,
             output_base_folder=output_base_folder,
             data_path=data_path, batch_size=batch_size, model_size=model_size, test=test)
