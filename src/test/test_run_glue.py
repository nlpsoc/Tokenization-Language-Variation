from unittest import TestCase
from run_glue import main
import sys


class Test(TestCase):
    def test_main(self):
        test_args = [
            "run_glue.py",
            "--model_name_or_path", "prajjwal1/bert-tiny",
            "--task_name", "mrpc",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--output_dir", "../output/"
        ]
        sys.argv = test_args
        main()
