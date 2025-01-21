import argparse
import ast
import json

from styletokenizer.utility.env_variables import set_cache

set_cache()

import os
import numpy as np
import pandas as pd

from datasets import DatasetDict
from styletokenizer.utility.umich_av import create_sadiri_class_dataset

from styletokenizer.tokenizer import TOKENIZER_PATHS
from styletokenizer.glue import GLUE_TASKS, GLUE_TEXTFLINT_TASKS, GLUE_TEXTFLINT, GLUE_MVALUE_TASKS, GLUE_MVALUE
from run_glue import task_to_keys as glue_task_to_keys
from styletokenizer.utility.datasets_helper import (
    load_data, VARIETIES_DEV_DICT, VARIETIES_TRAIN_DICT,
    VARIETIES_to_keys, VARIETIES_TASKS, VALUE_PATHS,
    VARIETIES_to_labels, VARIETIES_TEST_DICT
)
from styletokenizer.utility.tokenizer_vars import get_tokenizer_from_path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, classification_report, accuracy_score, hamming_loss
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier


def parse_label_if_str(example, label="label"):
    label_val = example[label]
    if isinstance(label_val, str):
        try:
            label_val = ast.literal_eval(label_val)
        except:
            label_val = [label_val]  # fallback
    example[label] = label_val
    return example


def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key):
    if sentence2_key is None:
        # Tokenize single sentence
        encodings1 = [tokenizer.encode(text) for text in examples[sentence1_key] if text is not None]
        input_ids1 = [encoding.ids for encoding in encodings1]
        return {'input_ids1': input_ids1}
    else:
        # Tokenize both sentences separately
        encodings = [
            [tokenizer.encode(text1), tokenizer.encode(text2)]
            for text1, text2 in zip(examples[sentence1_key], examples[sentence2_key])
            if text1 is not None and text2 is not None
        ]
        input_ids1 = [encoding[0].ids for encoding in encodings]
        input_ids2 = [encoding[1].ids for encoding in encodings]
        return {'input_ids1': input_ids1, 'input_ids2': input_ids2}


def ids_to_tokens(input_ids_list, tokenizer):
    # Convert list of input IDs to list of token strings
    return [" ".join([tokenizer.id_to_token(id_) for id_ in ids]) for ids in input_ids_list]


def get_common_words_features(tokens1_list, tokens2_list):
    # Generate features for common words
    features = []
    for tokens1, tokens2 in zip(tokens1_list, tokens2_list):
        tokens1_set = set(tokens1.split())
        tokens2_set = set(tokens2.split())
        common_tokens = tokens1_set & tokens2_set
        unique_tokens = (tokens1_set | tokens2_set) - common_tokens
        # Create 'word_word' features
        feature_tokens = [f"{word}_{word}" for word in common_tokens]
        uncommon_tokens = [f"{word}_not_{word}" for word in unique_tokens]
        features.append(' '.join(feature_tokens + uncommon_tokens))
    return features


def get_cross_words_features(tokens1_list, tokens2_list, max_tokens=50):
    # Generate features for cross words
    features = []
    for tokens1, tokens2 in zip(tokens1_list, tokens2_list):
        cross_tokens = [
            f"{w1}_{w2}"
            for w1 in tokens1.split()[:max_tokens]
            for w2 in tokens2.split()[:max_tokens]
        ]
        features.append(' '.join(cross_tokens))
    return features


def convert_multilabel_to_binary_matrix(label_lists):
    """
    Takes a list of lists (each sub-list = set of labels for one sample)
    and returns:
       - Y_binary: 2D array [num_samples, num_unique_labels] with 0/1
       - all_labels: the sorted list of unique labels
       - label_to_idx: mapping from label to column index
    """
    # Get unique labels
    all_labels = set()
    for label_list in label_lists:
        for lbl in label_list:
            all_labels.add(lbl)
    all_labels = sorted(list(all_labels))
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    # Create multi-hot matrix
    Y_binary = np.zeros((len(label_lists), len(all_labels)), dtype=int)
    for i, label_list in enumerate(label_lists):
        for lbl in label_list:
            Y_binary[i, label_to_idx[lbl]] = 1
    return Y_binary, all_labels, label_to_idx


def main(tasks="all", tokenizer_paths='all', on_test_set=False):
    features_type = 'cross_words'  # or 'common_words', etc.

    result_dict = {
        "task_name": [],
        "tokenizer_path": [],
        "F1_weighted": [],
        "F1_macro": [],
        "Accuracy_or_ExactMatch": [],
        "predictive_features": []
    }

    if tasks == "all":
        tasks = GLUE_MVALUE_TASKS + GLUE_TEXTFLINT_TASKS + VARIETIES_TASKS + GLUE_TASKS
    if tokenizer_paths == 'all':
        tokenizer_paths = [tok_path for tok_list in TOKENIZER_PATHS for tok_path in tok_list]

    if on_test_set:  # no testing data for GLUE publicly available
        testing_dict = VARIETIES_TEST_DICT
    else:
        testing_dict = VARIETIES_DEV_DICT

    for task_name_or_hfpath in tasks:
        task = task_name_or_hfpath

        if task in VARIETIES_TASKS:
            if task == "stel":  # not a training task
                continue
            if task_name_or_hfpath not in testing_dict.keys():
                raise ValueError(f"Task {task_name_or_hfpath} is not in testing set.")

            task_name_or_hfpath = testing_dict[task_name_or_hfpath]
            task_to_keys = VARIETIES_to_keys
            sentence_keys = task_to_keys[task]

            # Load dataset
            if task != "sadiri" and task != "PAN": #  and task != "NUCLE":
                raw_datasets = DatasetDict({
                    "train": load_data(csv_file=VARIETIES_TRAIN_DICT[task])["validation"],
                    "validation": load_data(csv_file=task_name_or_hfpath)["validation"]
                })
                print(f"loaded {task} from csv files {VARIETIES_TRAIN_DICT[task]} and {task_name_or_hfpath}")
            else:
                # Some tasks override the features_type to 'common_words'
                features_type = 'common_words'
                val_csv_path = testing_dict[task]
                raw_datasets = DatasetDict({
                    "train": load_data(csv_file=VARIETIES_TRAIN_DICT[task])["validation"],
                    "validation": load_data(csv_file=val_csv_path)["validation"]
                })
                print(f"loaded {task} from csv files {VARIETIES_TRAIN_DICT[task]} and {task_name_or_hfpath}")
        elif task_name_or_hfpath in GLUE_TEXTFLINT_TASKS:
            task = task_name_or_hfpath
            raw_datasets = DatasetDict({
                "train": load_data(csv_file=GLUE_TEXTFLINT[task_name_or_hfpath]["train"])["validation"],
                "validation": load_data(csv_file=GLUE_TEXTFLINT[task_name_or_hfpath]["dev"])["validation"]
            })
            sentence_keys = glue_task_to_keys[task_name_or_hfpath.split("-")[0]]
        elif task_name_or_hfpath in GLUE_MVALUE_TASKS:
            task = task_name_or_hfpath
            raw_datasets = DatasetDict({
                "train": load_data(csv_file=GLUE_MVALUE[task_name_or_hfpath]["train"])["validation"],
                "validation": load_data(csv_file=GLUE_MVALUE[task_name_or_hfpath]["dev"])["validation"]
            })
            sentence_keys = glue_task_to_keys[task_name_or_hfpath.split("-")[0]]
        elif task_name_or_hfpath in GLUE_TASKS:
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))
            task_to_keys = glue_task_to_keys
            raw_datasets = load_data(task_name_or_hfpath)
            print(f"loaded {task} from hf dataset {task_name_or_hfpath}")
            sentence_keys = task_to_keys[task]
        else:
            raise ValueError(f"Task {task_name_or_hfpath} not recognized.")

        # get label key
        label = "label"
        if task in VARIETIES_to_labels.keys():
            label = VARIETIES_to_labels[task]
        print(f"Task: {task}, label: {label}")

        raw_datasets = raw_datasets.map(lambda x: parse_label_if_str(x, label))

        val_key = "validation_matched" if task == "mnli" else "validation"

        def filter_none_labels(example, sentence1_key, sentence2_key=None):
            if sentence2_key is None:
                return example[sentence1_key] is not None and example[label] is not None
            else:
                return (
                        example[sentence1_key] is not None
                        and example[sentence2_key] is not None
                        and example[label] is not None
                )

        # Filter out None labels
        raw_datasets['train'] = raw_datasets['train'].filter(
            lambda x: filter_none_labels(x, sentence_keys[0], sentence_keys[1] if len(sentence_keys) > 1 else None)
        )
        raw_datasets[val_key] = raw_datasets[val_key].filter(
            lambda x: filter_none_labels(x, sentence_keys[0], sentence_keys[1] if len(sentence_keys) > 1 else None)
        )

        for tokenizer_path in tokenizer_paths:
            tokenizer = get_tokenizer_from_path(tokenizer_path)

            out_path = f"{os.path.dirname(tokenizer_path)}/LR/{task}"
            os.makedirs(out_path, exist_ok=True)

            # Preprocess / tokenize
            encoded_dataset = raw_datasets.map(
                lambda examples: preprocess_function(
                    examples, tokenizer, sentence_keys[0],
                    sentence_keys[1] if len(sentence_keys) > 1 else None
                ),
                batched=True,
                load_from_cache_file=False,  # re-tokenize every time
            )

            # Extract labels
            y_train = encoded_dataset["train"][label]
            y_eval = encoded_dataset[val_key][label]

            # -------------------------------
            #   1) Build text features
            # -------------------------------
            if (len(sentence_keys) == 1) or (sentence_keys[1] is None):
                # Single-sentence tasks
                X_train_ids1 = encoded_dataset["train"]["input_ids1"]
                X_eval_ids1 = encoded_dataset[val_key]["input_ids1"]

                X_train_texts = ids_to_tokens(X_train_ids1, tokenizer)
                X_eval_texts = ids_to_tokens(X_eval_ids1, tokenizer)

            else:
                # Sentence-pair tasks
                X_train_ids1 = encoded_dataset["train"]["input_ids1"]
                X_train_ids2 = encoded_dataset["train"]["input_ids2"]
                X_eval_ids1 = encoded_dataset[val_key]["input_ids1"]
                X_eval_ids2 = encoded_dataset[val_key]["input_ids2"]

                X_train_tokens1 = ids_to_tokens(X_train_ids1, tokenizer)
                X_train_tokens2 = ids_to_tokens(X_train_ids2, tokenizer)
                X_eval_tokens1 = ids_to_tokens(X_eval_ids1, tokenizer)
                X_eval_tokens2 = ids_to_tokens(X_eval_ids2, tokenizer)

                if features_type == 'common_words':
                    X_train_texts = get_common_words_features(X_train_tokens1, X_train_tokens2)
                    X_eval_texts = get_common_words_features(X_eval_tokens1, X_eval_tokens2)
                elif features_type == 'cross_words':
                    X_train_texts = get_cross_words_features(X_train_tokens1, X_train_tokens2)
                    X_eval_texts = get_cross_words_features(X_eval_tokens1, X_eval_tokens2)
                else:
                    raise ValueError("Invalid features_type. Choose 'common_words' or 'cross_words'.")

            # Bag-of-Words vectorizer
            vectorizer = CountVectorizer()
            X_train_features = vectorizer.fit_transform(X_train_texts)
            X_eval_features = vectorizer.transform(X_eval_texts)

            # -------------------------------------------------
            #   2) Check if multi-label (labels are lists?)
            # -------------------------------------------------
            # A quick check: if first sample's label is a list, assume multi-label
            is_multilabel = False
            if len(y_train) > 0 and isinstance(y_train[0], list):
                is_multilabel = True

            # -------------------------------------------------
            #   3) Training and evaluation
            # -------------------------------------------------
            if is_multilabel:
                print(f"\n[INFO] Detected multi-label data for task: {task} with tokenizer {tokenizer_path}")
                # Convert label lists to multi-hot vectors
                y_train_binary, all_labels, label_to_idx = convert_multilabel_to_binary_matrix(y_train)
                y_eval_binary, _, _ = convert_multilabel_to_binary_matrix(y_eval)

                # Train OneVsRest logistic regression for multi-label
                clf = OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver='liblinear')
                )
                clf.fit(X_train_features, y_train_binary)

                # Predict
                y_pred_binary = clf.predict(X_eval_features)

                # Evaluate: multi-label
                f1_weighted = f1_score(y_eval_binary, y_pred_binary, average='weighted', zero_division=0)
                f1_macro = f1_score(y_eval_binary, y_pred_binary, average='macro', zero_division=0)
                f1_per_label = f1_score(y_eval_binary, y_pred_binary, average=None, zero_division=0)

                # Print F1 score per label
                for label, f1 in zip(all_labels, f1_per_label):
                    print(f"F1 score for label {label}: {f1:.4f}")

                # save F1 per label in text and the weighted, macro F1 scores
                with open(f"{out_path}/f1_per_label.txt", "w") as f:
                    f.write(f"F1 per label: {f1_per_label}\n")
                    f.write(f"F1 weighted: {f1_weighted}\n")
                    f.write(f"F1 macro: {f1_macro}\n")

                # Subset accuracy (exact match ratio)
                # fraction of samples that have ALL labels correct
                exact_match_ratio = (y_pred_binary == y_eval_binary).all(axis=1).mean()

                # Record
                result_dict["task_name"].append(task)
                result_dict["tokenizer_path"].append(tokenizer_path)
                result_dict["F1_weighted"].append(f1_weighted)
                result_dict["F1_macro"].append(f1_macro)
                result_dict["Accuracy_or_ExactMatch"].append(exact_match_ratio)

                # ---------------------------
                #   Predictive features
                # ---------------------------
                # Each label has its own estimator in OneVsRestClassifier
                # If it's strictly binary for each label, .coef_.shape => (1, #features)
                # We'll store top +/- for each label
                feature_names = vectorizer.get_feature_names_out()
                top_features_for_all = {}

                # Each estimator_ is the logistic regression for that label
                for i, label_name in enumerate(all_labels):
                    lr_estimator = clf.estimators_[i]  # logistic regression model
                    # For binary (only 1 class in .classes_ might be odd if data is degenerate)
                    if len(lr_estimator.classes_) == 2:
                        # shape is (2, n_features)
                        coefs = lr_estimator.coef_[0]  # we only look at index 0
                    else:
                        coefs = lr_estimator.coef_[0]

                    # Sort by coefficient magnitude
                    sorted_indices = np.argsort(coefs)
                    top_positive_indices = sorted_indices[-100:][::-1]  # top 100
                    top_negative_indices = sorted_indices[:100]

                    top_features_for_all[label_name] = {
                        "top_positive": [
                            (feature_names[idx], coefs[idx]) for idx in top_positive_indices
                        ],
                        "top_negative": [
                            (feature_names[idx], coefs[idx]) for idx in top_negative_indices
                        ]
                    }

                result_dict["predictive_features"].append(top_features_for_all)
                # save top features as dict to out_path
                with open(f"{out_path}/top_features.json", "w") as f:
                    json.dump({str(k): v for k, v in top_features_for_all.items()}, f)

                print(f"Multi-label Classification Scores for {task} w/ {tokenizer_path}:")
                print(f"  F1 (weighted): {f1_weighted:.4f}")
                print(f"  F1 (macro): {f1_macro:.4f}")
                print(f"  Exact Match Ratio: {exact_match_ratio:.4f}")
                print(f"Top features (per label): {top_features_for_all}")
                print("------------------------------------------------------")

            else:
                # ------------------------------
                # Single-label classification
                # ------------------------------
                print(f"\n[INFO] Single-label data for task: {task} with tokenizer {tokenizer_path}")
                clf = LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver='liblinear')
                clf.fit(X_train_features, y_train)

                y_pred = clf.predict(X_eval_features)

                f1_weighted = f1_score(y_eval, y_pred, average='weighted')
                f1_macro = f1_score(y_eval, y_pred, average='macro')
                accuracy = accuracy_score(y_eval, y_pred)

                # Record
                result_dict["task_name"].append(task)
                result_dict["tokenizer_path"].append(tokenizer_path)
                result_dict["F1_weighted"].append(f1_weighted)
                result_dict["F1_macro"].append(f1_macro)
                result_dict["Accuracy_or_ExactMatch"].append(accuracy)

                # Identify most predictive features (similar to original code)
                feature_names = vectorizer.get_feature_names_out()
                coefs = clf.coef_

                # save coefficients to out_path
                np.save(f"{out_path}/coefs.npy", coefs)

                # If binary classification, stack negative & positive
                if len(clf.classes_) == 2:
                    coefs = np.vstack([-coefs[0], coefs[0]])
                elif len(coefs.shape) == 1:
                    coefs = coefs.reshape(1, -1)

                top_features = {}
                for i, class_label in enumerate(clf.classes_):
                    coef = coefs[i]
                    non_zero_indices = np.flatnonzero(coef)
                    if non_zero_indices.size == 0:
                        top_features[class_label] = {
                            "top_positive": [],
                            "top_negative": []
                        }
                        continue

                    sorted_indices = np.argsort(coef[non_zero_indices])
                    top_positive_indices = non_zero_indices[sorted_indices][-100:][::-1]
                    top_negative_indices = non_zero_indices[sorted_indices][:100]

                    top_features[class_label] = {
                        "top_positive": [(feature_names[j], coef[j]) for j in top_positive_indices],
                        "top_negative": [(feature_names[j], coef[j]) for j in top_negative_indices]
                    }

                # save top features as dict to out_path
                with open(f"{out_path}/top_features.json", "w") as f:
                    json.dump({str(k): v for k, v in top_features.items()}, f)

                result_dict["predictive_features"].append(top_features)

                # Print classification report for single-label
                print(f"Classification Report for {task} with tokenizer {tokenizer_path}:")
                print(classification_report(y_eval, y_pred))
                print("------------------------------------------------------")
                print(f"Predictive Features for {task} with tokenizer {tokenizer_path}:")
                print(top_features)

                # save Classification report
                with open(f"{out_path}/classification_report.txt", "w") as f:
                    f.write(classification_report(y_eval, y_pred))


            # save results
            result_df = pd.DataFrame(result_dict)


    # -------------------------
    #   Print and Save Results
    # -------------------------
    for idx in range(len(result_dict["task_name"])):
        print(f"Task: {result_dict['task_name'][idx]}")
        print(f"Tokenizer: {result_dict['tokenizer_path'][idx]}")
        print(f"F1 Weighted: {result_dict['F1_weighted'][idx]}")
        print(f"F1 Macro: {result_dict['F1_macro'][idx]}")
        print(f"Accuracy/Exact-Match: {result_dict['Accuracy_or_ExactMatch'][idx]}")
        print(f"Predictive Features: {result_dict['predictive_features'][idx]}")
        print("======================================================")

    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("logreg_eval_results.csv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", help="task to evaluate")
    parser.add_argument("--tokenizer_paths", type=str, default="all", help="tokenizer paths to evaluate")
    parser.add_argument("--on_test_set", action="store_true", help="evaluate on test set")
    args = parser.parse_args()

    tasks = args.task
    if tasks != "all":
        tasks = tasks.split(",")
    tokenizer_paths = args.tokenizer_paths
    if tokenizer_paths != "all":
        tokenizer_paths = tokenizer_paths.split(",")

    main(tasks=tasks, tokenizer_paths=tokenizer_paths, on_test_set=args.on_test_set)
