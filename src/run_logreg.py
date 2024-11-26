import os
from styletokenizer.utility.env_variables import set_cache

set_cache()

from styletokenizer.tokenizer import TOKENIZER_PATHS
from styletokenizer.glue import GLUE_TASKS
from run_glue import task_to_keys as glue_task_to_keys
from styletokenizer.utility.datasets_helper import load_data
from styletokenizer.utility.tokenizer_vars import get_tokenizer_from_path

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key):
    if sentence2_key is None:
        # Tokenize single sentence
        encodings1 = [tokenizer.encode(text) for text in examples[sentence1_key]]
        input_ids1 = [encoding.ids for encoding in encodings1]
        return {'input_ids1': input_ids1}
    else:
        # Tokenize both sentences separately
        encodings1 = [tokenizer.encode(text1) for text1 in examples[sentence1_key]]
        encodings2 = [tokenizer.encode(text2) for text2 in examples[sentence2_key]]
        input_ids1 = [encoding.ids for encoding in encodings1]
        input_ids2 = [encoding.ids for encoding in encodings2]
        return {'input_ids1': input_ids1, 'input_ids2': input_ids2}


def ids_to_tokens(input_ids_list, tokenizer):
    # Convert list of input IDs to list of token strings
    return [" ".join([tokenizer.id_to_token(id) for id in ids]) for ids in input_ids_list]


def get_common_words_features(tokens1_list, tokens2_list):
    # Generate features for common words
    features = []
    for tokens1, tokens2 in zip(tokens1_list, tokens2_list):
        tokens1_set = set(tokens1.split())
        tokens2_set = set(tokens2.split())
        common_tokens = tokens1_set & tokens2_set
        # Create 'word_word' features
        feature_tokens = [f"{word}_{word}" for word in common_tokens]
        features.append(' '.join(feature_tokens))
    return features


def get_cross_words_features(tokens1_list, tokens2_list, max_tokens=50):
    # Generate features for cross words
    features = []
    for tokens1, tokens2 in zip(tokens1_list, tokens2_list):
        # Create Cartesian product of tokens
        cross_tokens = [f"{w1}_{w2}" for w1 in tokens1.split()[:max_tokens] for w2 in tokens2.split()[:max_tokens]]
        features.append(' '.join(cross_tokens))
    return features


def main():
    features_type = 'cross_words'  # Change to 'common_words' as needed

    result_dict = {
        "task_name": [],
        "tokenizer_path": [],
        "F1_weighted": [],
        "F1_macro": [],
        "Accuracy": [],
        "predictive_features": []
    }

    for task_name_or_hfpath in ["snli"] + GLUE_TASKS:
        task = os.path.basename(os.path.normpath(task_name_or_hfpath))
        raw_datasets = load_data(task_name_or_hfpath)

        task_to_keys = glue_task_to_keys[task]
        sentence1_key, sentence2_key = task_to_keys
        val_key = "validation_matched" if task == "mnli" else "validation"

        for tokenizer_path in TOKENIZER_PATHS:
            tokenizer = get_tokenizer_from_path(tokenizer_path)

            # Use lambda to pass tokenizer and sentence keys to preprocess_function
            encoded_dataset = raw_datasets.map(
                lambda examples: preprocess_function(examples, tokenizer, sentence1_key, sentence2_key),
                batched=True,
                load_from_cache_file=False  # Make sure to re-tokenize every time
            )

            # Extract labels
            y_train = encoded_dataset["train"]["label"]
            y_eval = encoded_dataset[val_key]["label"]

            if sentence2_key is None:
                # Single sentence tasks
                X_train_ids1 = encoded_dataset["train"]["input_ids1"]
                X_eval_ids1 = encoded_dataset[val_key]["input_ids1"]

                # Convert input IDs to tokens
                X_train_texts = ids_to_tokens(X_train_ids1, tokenizer)
                X_eval_texts = ids_to_tokens(X_eval_ids1, tokenizer)

            else:
                # Sentence pair tasks
                X_train_ids1 = encoded_dataset["train"]["input_ids1"]
                X_train_ids2 = encoded_dataset["train"]["input_ids2"]
                X_eval_ids1 = encoded_dataset[val_key]["input_ids1"]
                X_eval_ids2 = encoded_dataset[val_key]["input_ids2"]

                # Convert input IDs to tokens
                X_train_tokens1 = ids_to_tokens(X_train_ids1, tokenizer)
                X_train_tokens2 = ids_to_tokens(X_train_ids2, tokenizer)
                X_eval_tokens1 = ids_to_tokens(X_eval_ids1, tokenizer)
                X_eval_tokens2 = ids_to_tokens(X_eval_ids2, tokenizer)

                if features_type == 'common_words':
                    # Create features based on common words
                    X_train_texts = get_common_words_features(X_train_tokens1, X_train_tokens2)
                    X_eval_texts = get_common_words_features(X_eval_tokens1, X_eval_tokens2)
                elif features_type == 'cross_words':
                    # Create features based on cross words
                    X_train_texts = get_cross_words_features(X_train_tokens1, X_train_tokens2)
                    X_eval_texts = get_cross_words_features(X_eval_tokens1, X_eval_tokens2)
                else:
                    raise ValueError("Invalid features_type. Choose 'common_words' or 'cross_words'.")

            # Create Bag-of-Words features
            vectorizer = CountVectorizer()
            X_train_features = vectorizer.fit_transform(X_train_texts)
            X_eval_features = vectorizer.transform(X_eval_texts)

            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, penalty="l1", C=0.4, solver='liblinear')
            print(f"Training logistic regression for {task} with tokenizer {tokenizer_path}")
            clf.fit(X_train_features, y_train)

            # Predict on evaluation set
            y_pred = clf.predict(X_eval_features)

            # Compute evaluation metrics
            f1_weighted = f1_score(y_eval, y_pred, average='weighted')
            f1_macro = f1_score(y_eval, y_pred, average='macro')
            accuracy = accuracy_score(y_eval, y_pred)

            # Record results
            result_dict["task_name"].append(task)
            result_dict["tokenizer_path"].append(tokenizer_path)
            result_dict["F1_weighted"].append(f1_weighted)
            result_dict["F1_macro"].append(f1_macro)
            result_dict["Accuracy"].append(accuracy)

            # Identify most predictive features
            feature_names = vectorizer.get_feature_names_out()
            coefs = clf.coef_
            if len(coefs.shape) == 1:
                coefs = [coefs]
            top_features = {}
            for i, class_label in enumerate(clf.classes_):
                top_positive_coefficients = np.argsort(coefs[i])[-10:]
                top_negative_coefficients = np.argsort(coefs[i])[:10]
                top_features[class_label] = {
                    "top_positive": [feature_names[j] for j in top_positive_coefficients],
                    "top_negative": [feature_names[j] for j in top_negative_coefficients]
                }
            result_dict["predictive_features"].append(top_features)

            # Optionally, print classification report
            print(f"Classification Report for {task} with tokenizer {tokenizer_path}:")
            print(classification_report(y_eval, y_pred))
            print("------------------------------------------------------")

    # Print the results
    for idx in range(len(result_dict["task_name"])):
        print(f"Task: {result_dict['task_name'][idx]}")
        print(f"Tokenizer: {result_dict['tokenizer_path'][idx]}")
        print(f"F1 Weighted: {result_dict['F1_weighted'][idx]}")
        print(f"F1 Macro: {result_dict['F1_macro'][idx]}")
        print(f"Accuracy: {result_dict['Accuracy'][idx]}")
        print(f"Predictive Features: {result_dict['predictive_features'][idx]}")
        print("======================================================")

    # Save the results
    import pandas as pd
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv("logreg_eval_results.csv", sep="\t")


if __name__ == "__main__":
    main()
