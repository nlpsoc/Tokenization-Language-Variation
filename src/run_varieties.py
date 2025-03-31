"""
    run tasks sensitive to language variation

"""
import os
import argparse
import subprocess
from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.sensitive_tasks import VARIETIES_DEV_DICT, VARIETIES_TRAIN_DICT, VARIETIES_to_keys, \
    VARIETIES_to_labels

tasks = ["sadiri", "stel", "age", "mrpc", "sst2", "qqp", "mnli", "qnli", "rte", "CORE", "CGLU", "GYAFC", "DIALECT",
         "SNLI-NLI", "SNLI-Style", "SNLI", "convo-style", "NUCLE", "PAN", "simplification", "multi-DIALECT"]


def main(task, model_path, seed, output_dir, overwrite=False):
    # print all set vars
    log_and_flush(f"task: {task}")
    log_and_flush(f"model_path: {model_path}")
    log_and_flush(f"seed: {seed}")
    log_and_flush(f"output_dir: {output_dir}")

    if task == "sadiri":
        out_model_path = os.path.join(output_dir, "best_model")
        # only call calculation if model does not exist
        if not os.path.exists(out_model_path):
            log_and_flush("Model does not exist, training.")
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
        else:
            log_and_flush("Model already exists, skipping training.")
        # then, on validation set, load best model and find the best threshold for cosine similarities
        # load model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(out_model_path)

        # get threshold on validation set
        import pandas as pd
        from torch.nn.functional import cosine_similarity
        import numpy as np
        validation_csv_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation/validation.csv"
        validation_df = pd.read_csv(validation_csv_path)
        query_embeddings = model.encode(validation_df["query_text"].tolist(), convert_to_tensor=True)
        candidate_embeddings = model.encode(validation_df["candidate_text"].tolist(), convert_to_tensor=True)
        similarities = cosine_similarity(query_embeddings, candidate_embeddings)

        y_true = validation_df["label"].values
        best_threshold = None
        best_accuracy = 0.0

        for t in np.linspace(0, 1, 101):  # 0.0, 0.01, 0.02, ... 1.0
            y_pred = (similarities >= t).long().cpu().numpy()  # or .astype(int)
            accuracy = (y_true == y_pred).mean()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        log_and_flush(f"Best threshold: {best_threshold}")
        log_and_flush(f"Best accuracy on validation set: {best_accuracy}")

        # calculate accuracy on test set
        test_csv_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/test/test.csv"
        if not os.path.exists(test_csv_path):  # only called the once, in parallel for the train/val split
            test_folder = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/test"
            from styletokenizer.utility.umich_av import create_singplesplit_sadiri_classification_dataset
            test_df = create_singplesplit_sadiri_classification_dataset(test_folder)
            test_df.to_csv(test_csv_path, index=False)
        test_df = pd.read_csv(test_csv_path)

        query_embeddings = model.encode(test_df["query_text"].tolist(), convert_to_tensor=True)
        candidate_embeddings = model.encode(test_df["candidate_text"].tolist(), convert_to_tensor=True)
        similarities = cosine_similarity(query_embeddings, candidate_embeddings)
        y_true = test_df["label"].values

        y_pred = (similarities >= best_threshold).long().cpu().numpy()
        accuracy = (y_true == y_pred).mean()

        log_and_flush(f"Test accuracy: {accuracy}")
        # save accuracy in a txt file in the output_dir
        with open(os.path.join(output_dir, "test_accuracy.txt"), "w") as f:
            f.write(f"Best threshold: {best_threshold}\n")
            f.write(f"Best accuracy on validation set: {best_accuracy}\n")
            f.write(f"Test accuracy: {accuracy}\n")

        # save predictions
        test_df["predictions"] = y_pred
        test_df.to_csv(os.path.join(output_dir, "eval_dataset.tsv"), index=False, sep="\t")
        # print accuracy on test set
        # train_csv_path = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train/train.csv"
        # from styletokenizer.utility.umich_av import create_singplesplit_sadiri_classification_dataset
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
    elif task == "PAN":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", VARIETIES_TRAIN_DICT["PAN"],
            "--validation_file", VARIETIES_DEV_DICT["PAN"],
            "--shuffle_train_dataset",
            "--text_column_name", "text 1,text 2",
            "--text_column_delimiter", "[SEP]",
            "--label_column_name", VARIETIES_to_labels["PAN"],
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
    elif task == "NUCLE":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", VARIETIES_TRAIN_DICT["NUCLE"],
            "--validation_file", VARIETIES_DEV_DICT["NUCLE"],
            "--shuffle_train_dataset",
            "--text_column_name", VARIETIES_to_keys["NUCLE"][0],  # "sentence1,sentence2",
            "--label_column_name", VARIETIES_to_labels["NUCLE"],
            "--text_column_delimiter", "[SEP]",
            "--do_train",
            "--do_eval",
            "--max_seq_length", "128",
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
    elif task == "multi-DIALECT":
        command = [
            "python", "run_classification.py",
            "--model_name_or_path", model_path,
            "--train_file", VARIETIES_TRAIN_DICT[task],
            "--validation_file", VARIETIES_DEV_DICT[task],
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
