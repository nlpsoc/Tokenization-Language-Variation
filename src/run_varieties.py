import os
import argparse
import subprocess
from styletokenizer.utility.custom_logger import log_and_flush

tasks = ["sadiri", "stel", "age", "value", "CORE"]


def main(task, model_path, seed, output_dir):
    # print all set vars
    log_and_flush(f"task: {task}")
    log_and_flush(f"model_path: {model_path}")
    log_and_flush(f"seed: {seed}")
    log_and_flush(f"output_dir: {output_dir}")

    if task == "sadiri":
        command = [
            "python", "sadiri_main.py",
            "--train",
            "--validate",
            "--out_dir", output_dir,
            "--pretrained_model", model_path,
            "--train_data", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train",
            "--dev_data", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation",
            "--learning_rate", "0.00001",
            "--batch_size", "128",
            "--epochs", "5",
            "--max_length", "512",
            "--grad_acc", "1",
            "--gradient_checkpointing", "False",
            "--saving_step", "100",
            "--mask", "0",
            "--seed", str(seed),
            "--corpus", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle",
            "--loss", "SupConLoss"
        ]
        result = subprocess.run(command)

    elif task == "stel":
        from styletokenizer.utility.env_variables import set_cache
        set_cache()
        import sys
        # add STEL folder to path
        sys.path.append('../../STEL/src/')
        import torch
        from STEL import STEL
        from STEL.similarity import Similarity, cosine_sim
        from sentence_transformers import SentenceTransformer

        class SBERTSimilarity(Similarity):
            def __init__(self):
                super().__init__()
                self.model = SentenceTransformer(model_path)
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")

            def similarities(self, sentences_1, sentences_2):
                with torch.no_grad():
                    sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
                    sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
                return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]

        STEL.eval_on_STEL(style_objects=[SBERTSimilarity()])
    elif task == "age":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus/train.csv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus/validation.csv",
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--remove_columns", "age,date,gender,horoscope,job",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--max_train_samples", "200000",  # use only 200k samples, which is roughly 10% of the dataset
            "--output_dir", output_dir,
            "--seed", str(seed),
        ]
        # REGRESSION VERSION
        # command = [
        #     "python", "run_classification.py",
        #     "--model_name_or_path", model_path,
        #     "--dataset_name", 'barilan/blog_authorship_corpus',
        #     "--trust_remote_code", "True",
        #     "--shuffle_train_dataset",
        #     "--do_regression", "True",
        #     "--metric_name", "mse",  # "mse" is default for regression
        #     "--text_column_name", "text",
        #     "--label_column_name", "age"
        #     "--remove_columns", "date,gender,horoscope,job",
        #     "--do_train",
        #     "--do_eval",
        #     "--max_seq_length", "512",
        #     "--per_device_train_batch_size", "32",
        #     "--learning_rate", "2e-5",
        #     "--num_train_epochs", "3",
        #     "--max_train_samples", "200000",  # use only 200k samples, which is roughly 10% of the dataset
        #     "--output_dir", output_dir,
        #     "--seed", str(seed),
        # ]
        result = subprocess.run(command)
    elif task == "value":  # currently not in use, as value seems to need different code / env
        command = [
            "python", "run_value.py",
            "--model_name_or_path", model_path,
            "--dataset_name", 'barilan/blog_authorship_corpus',
            "--trust_remote_code", "True",
            "--shuffle_train_dataset",
            "--metric_name", "f1",
            "--text_column_name", "text",
            "--label_column_name", "value",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--seed", str(seed),
        ]
        result = subprocess.run(command)

    elif task == "CORE":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/multilabel_train.tsv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/multilabel_dev.tsv",
            # "--test_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/prepped_test.tsv",
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "5",
            "--output_dir", output_dir,
            "--seed", str(seed),
        ]
        result = subprocess.run(command)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', default='sadiri', type=str, choices=tasks)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()
    main(task=args.task, model_path=args.model_path, seed=args.seed, output_dir=args.output_dir)
