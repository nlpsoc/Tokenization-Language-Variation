import os
import argparse
import subprocess
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.datasets_helper import VARIETIES_TRAIN_DICT, VARIETIES_DEV_DICT

tasks = ["sadiri", "stel", "age", "mrpc", "sst2", "qqp", "mnli", "qnli", "rte", "CORE", "CGLU", "GYAFC", "DIALECT",
         "SNLI-NLI", "SNLI-Style", "SNLI", "convo-style", "NUCLE", "PAN"]


def main(task, model_path, seed, output_dir, overwrite=False):
    # print all set vars
    log_and_flush(f"task: {task}")
    log_and_flush(f"model_path: {model_path}")
    log_and_flush(f"seed: {seed}")
    log_and_flush(f"output_dir: {output_dir}")

    if task == "sadiri":
        # MRR calculation
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
        # then, on validation set, load best model and find the best threshold for cosine similarities


        # load model
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(output_dir, "best_model")
        model = SentenceTransformer(model_path)

        # get threshold on validation set
        import pandas as pd
        from torch.nn.functional import cosine_similarity
        import numpy as np
        validation_csv_path = ("/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation"
                               "/validation.csv")
        validation_df = pd.read_csv(validation_csv_path)
        query_embeddings = model.encode(validation_df["query_text"].tolist(), convert_to_tensor=True)
        candidate_embeddings = model.encode(validation_df["candidate_text"].tolist(), convert_to_tensor=True)
        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        y_true = validation_df["label"].values
        best_threshold = None
        best_accuracy = 0.0

        for t in np.linspace(0, 1, 101):  # 0.0, 0.01, 0.02, ... 1.0
            y_pred = (similarities >= t).long().numpy()  # or .astype(int)
            accuracy = (y_true == y_pred).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        print("Best threshold:", best_threshold)
        print("Best accuracy on validation set:", best_accuracy)

        # calculate accuracy on test set
        test_csv_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/test/test.csv"
        test_df = pd.read_csv(test_csv_path)

        query_embeddings = model.encode(test_df["query_text"].tolist(), convert_to_tensor=True)
        candidate_embeddings = model.encode(test_df["candidate_text"].tolist(), convert_to_tensor=True)
        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        y_pred = (similarities >= best_threshold).long().numpy()
        accuracy = (y_true == y_pred).mean()

        print("Test accuracy:", accuracy)
        # print accuracy on test set
        # train_csv_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train/train.csv"
        from styletokenizer.utility.umich_av import create_singplesplit_sadiri_classification_dataset
        # train_file = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train"
        # train_dataset = create_singplesplit_sadiri_classification_dataset(train_file)
        # train_dataset.to_csv(train_csv_path, index=False)
        # validation_file = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation"
        # validation_dataset = create_singplesplit_sadiri_classification_dataset(validation_file)
        # validation_dataset.to_csv(validation_csv_path, index=False)
        #
        # command = [
        #     "python", "run_classification.py",
        #     "--model_name_or_path", model_path,
        #     "--train_file", train_csv_path,
        #     "--validation_file", validation_csv_path,
        #     "--shuffle_train_dataset",
        #     "--text_column_name", "query_text,candidate_text",
        #     "--text_column_delimiter", "[SEP]",
        #     "--label_column_name", "label",
        #     "--do_train",
        #     "--do_eval",
        #     "--max_seq_length", "512",
        #     "--per_device_train_batch_size", "32",
        #     "--learning_rate", "2e-5",
        #     "--num_train_epochs", "3",
        #     # "--max_train_samples", "200000",
        #     "--output_dir", output_dir,
        #     "--seed", str(seed),
        #     "--overwrite_cache",
        #     # "--metric_name", "f1",
        #     "--save_strategy", "epoch",
        # ]
        # if overwrite:
        #     command.append("--overwrite_output_dir")

    elif task == "convo-style":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--dataset_name", "AnnaWegmann/StyleEmbeddingData",
            # "--shuffle_train_dataset",
            "--text_column_name", "Anchor (A),Utterance 1 (U1)",
            "--label_column_name", "Same Author Label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            # "--max_train_samples", "200000",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            # "--metric_name", "f1",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
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
                self.model = SentenceTransformer(
                    model_path)  # should create mean pooling by default, see https://www.sbert.net/docs/sentence_transformer/usage/custom_models.html?highlight=mean%20pool
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")

            def similarities(self, sentences_1, sentences_2):
                with torch.no_grad():
                    sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
                    sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
                return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]

        STEL.eval_on_STEL(style_objects=[SBERTSimilarity()])
    elif task == "age":
        command = [
            "python", "run_classification_org.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus/train_sampled.csv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus/validation_sampled.csv",
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--remove_columns", "age,date,gender,horoscope,job",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--max_train_samples", "100000",  # use only 200k samples, which is roughly 5% of the dataset
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
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
    elif task in ["mrpc", "sst2", "qqp", "mnli", "qnli",
                  "rte"]:  # currently not in use, as value seems to need different code / env
        if task == "mrpc":
            train_epochs = "5"
        else:
            train_epochs = "3"
        command = [
            "python", "run_glue.py",  # TODO: use the cleaner run_glue_old script
            "--model_name_or_path", model_path,
            "--task_name", f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/value/{task}/",
            "--trust_remote_code", "True",
            # "--shuffle_train_dataset",
            # "--text_column_name", "text",
            # "--label_column_name", "label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", train_epochs,
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)

    elif task == "CORE":
        command = [
            "python", "run_classification.py",  # TODO: use the cleaner run_classification_old script
            "--model_name_or_path", model_path,
            "--train_file", VARIETIES_TRAIN_DICT[task],
            "--validation_file", VARIETIES_DEV_DICT[task],
            # "--test_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/prepped_test.tsv",
            # "--shuffle_train_dataset",
            # "--max_train_samples", "100000",  # using longer seq length, so reduce samples --> only > 30k instances anyway
            "--text_column_name", "text",
            "--label_column_name", "genre",  # "label"
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "5",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "CGLU":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Varieties/CGLUv5.2/train.csv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Varieties/CGLUv5.2/dev.csv",
            # "--test_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Varieties/CGLUv5.2/test.csv",
            "--shuffle_train_dataset",
            "--text_column_name", "Text",
            "--label_column_name", "origin",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "512",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--max_train_samples", "200000",
            "--max_eval_samples", "20000",
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "GYAFC":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GYAFC/train.csv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GYAFC/dev.csv",
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "NUCLE":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", VARIETIES_TRAIN_DICT["NUCLE"],
            "--validation_file", VARIETIES_DEV_DICT["NUCLE"],
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "DIALECT":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Dialect/combined_train.csv",
            "--validation_file",
            "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Dialect/combined_validation.csv",
            "--shuffle_train_dataset",
            "--text_column_name", "text",
            "--label_column_name", "label",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "eopch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "SNLI-NLI":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/train.tsv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/dev.tsv",
            "--shuffle_train_dataset",
            "--text_column_name", "premise,hypothesis",
            "--label_column_name", "nli",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "1",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "SNLI-Style":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/train.tsv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/dev.tsv",
            "--shuffle_train_dataset",
            "--text_column_name", "premise,hypothesis",
            "--label_column_name", "style",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "1",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)
    elif task == "SNLI":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/train.tsv",
            "--validation_file", "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/SNLI_modified/dev.tsv",
            "--shuffle_train_dataset",
            "--text_column_name", "premise_original,hypothesis_original",
            "--label_column_name", "nli",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "1",
            "--output_dir", output_dir,
            "--seed", str(seed),
            "--overwrite_cache",
            "--save_strategy", "epoch",
        ]
        if overwrite:
            command.append("--overwrite_output_dir")
        result = subprocess.run(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', default='sadiri', type=str, choices=tasks)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output directory if it exists", default=False)

    args = parser.parse_args()
    main(task=args.task, model_path=args.model_path, seed=args.seed, output_dir=args.output_dir,
         overwrite=args.overwrite)
