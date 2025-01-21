"""
    calculate the correlations between BERT predictions and log regression / intrinsic measures
"""
import json
import os
import re
import statistics

import pandas as pd

from styletokenizer.glue import GLUE_TEXTFLINT_TASKS
from styletokenizer.tokenizer import TOKENIZER_PATHS

performance_keys = {
    "sst2": "eval_accuracy",
    "qqp": "eval_f1",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
}


def main():
    # do this only for the textflint tasks for now
    tasks = GLUE_TEXTFLINT_TASKS
    tokenizer_paths = TOKENIZER_PATHS

    unique_tokenizer_paths = set()
    for tokenizer_path in tokenizer_paths:
        for path in tokenizer_path:
            unique_tokenizer_paths.add(path)

    # collect the BERT performance scores of the tasks

    local_finder_addition = "/Users/anna/sftp_mount/hpc_disk/02-awegmann/"
    server_finder_addition = "/hpc/uu_cs_nlpsoc/02-awegmann/"

    BERT_PERFORMANCE = get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition)
    print(BERT_PERFORMANCE)

    STATS_BASE_PATH = os.path.join(local_finder_addition, "TOKENIZER/tokenizer/")
    LOG_REGRESSION = get_logreg_performances(tasks, unique_tokenizer_paths, STATS_BASE_PATH)
    print(LOG_REGRESSION)
    # calculate the correlation between the BERT performance and the logistic regression
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION)
    print(f"Correlation between BERT and LR: {c}")

    # get intrinisic measures
    intrinsic_key = "avg_seq_len"  # renyi_eff_2.5
    INTRINSIC_MEASURE = method_name(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    # calculate the correlation between the BERT performance and the intrinsic measures
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE)
    print(f"Correlation between BERT and seq len: {c}")

    intrinsic_key = "renyi_eff_2.5"
    INTRINSIC_MEASURE = method_name(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE)
    print(f"Correlation between BERT and renyi eff 2.5: {c}")



def method_name(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH):
    INTRINSIC_MEASURE = {}
    intrinsic_addition = "intrinsic"
    for task in tasks:
        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            if tokenizer_name not in INTRINSIC_MEASURE:
                INTRINSIC_MEASURE[tokenizer_name] = {}
            # get the BERT output for the task
            result_path = os.path.join(STATS_BASE_PATH, tokenizer_name, intrinsic_addition, task, "eval_results.csv")
            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue
            # read in csv file with pandas
            df = pd.read_csv(result_path)
            # get the performance from the performance keys
            INTRINSIC_MEASURE[tokenizer_name][task] = df[intrinsic_key].values[0]
    return INTRINSIC_MEASURE


def calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION):
    x = []
    y = []
    for tokenizer_name in BERT_PERFORMANCE:
        for task in BERT_PERFORMANCE[tokenizer_name]:
            if task in LOG_REGRESSION[tokenizer_name]:
                x.append(BERT_PERFORMANCE[tokenizer_name][task])
                y.append(LOG_REGRESSION[tokenizer_name][task])
    return statistics.correlation(x, y)


def get_logreg_performances(tasks, unique_tokenizer_paths, stats_base_path):
    # collect the logistic regression
    LOG_REGRESSION = {}
    LR_ADDITION = "LR"
    for task in tasks:
        task_key = task
        if task in GLUE_TEXTFLINT_TASKS:
            task_key = task.split('-')[0]
        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            if tokenizer_name not in LOG_REGRESSION:
                LOG_REGRESSION[tokenizer_name] = {}
            # get the BERT output for the task
            result_path = os.path.join(stats_base_path, tokenizer_name, LR_ADDITION, task, "classification_report.txt")

            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue
            classification_report = parse_classification_report(result_path)

            # get the performance from the performance keys
            if "accuracy" in performance_keys[task_key]:
                LOG_REGRESSION[tokenizer_name][task] = classification_report["accuracy"]
            else:
                LOG_REGRESSION[tokenizer_name][task] = classification_report['1']["f1-score"]
    return LOG_REGRESSION


def get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition):
    BERT_PERFORMANCE = {}
    GLUE_OUT_BASE_PATH = os.path.join(local_finder_addition, "TOKENIZER/output/GLUE/textflint/base-BERT/")
    BERT_PATH = "749M/steps-45000/seed-42/42/"
    for task in tasks:
        task_key = task
        if task in GLUE_TEXTFLINT_TASKS:
            task_key = task.split('-')[0]

        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            if tokenizer_name not in BERT_PERFORMANCE:
                BERT_PERFORMANCE[tokenizer_name] = {}
            # get the BERT output for the task
            result_path = os.path.join(GLUE_OUT_BASE_PATH, tokenizer_name, BERT_PATH, task_key, "all_results.json")
            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue

            # read in json file
            with open(result_path, "r") as f:
                data_dict = json.load(f)
            # get the performance from the performance keys
            BERT_PERFORMANCE[tokenizer_name][task] = data_dict[performance_keys[task_key]]
    return BERT_PERFORMANCE


def parse_classification_report(file_path):
    """
    Parse a text-based classification report (such as the output of
    sklearn.metrics.classification_report) into a structured dictionary.

    This version explicitly captures accuracy as well.
    """

    # We'll store metrics in a dictionary structure like:
    # {
    #   "0": {"precision": 0.71, "recall": 0.67, "f1-score": 0.69, "support": 422},
    #   "1": {...},
    #   "accuracy": 0.71,                # separate float
    #   "macro avg": {...},
    #   "weighted avg": {...}
    # }

    parsed_report = {}

    with open(file_path, 'r') as f:
        # Read all lines, stripping whitespace, skipping empty lines
        lines = [line.strip() for line in f if line.strip()]

    # This regex will split on 2 or more spaces
    splitter = re.compile(r"\s{2,}")

    for line in lines:
        # Split the line by 2+ spaces.
        parts = splitter.split(line)

        # Typical lines might be:
        # ["0", "0.71", "0.67", "0.69", "422"]
        # ["1", "0.70", "0.74", "0.72", "443"]
        # ["accuracy", "0.71", "865"]                  # 3 columns
        # ["macro avg", "0.71", "0.71", "0.71", "865"]  # 5 columns
        # ["weighted avg", "0.71", "0.71", "0.71", "865"]

        if len(parts) == 5:
            # Likely a class label (e.g. "0", "1") or an avg (e.g. "macro avg", "weighted avg")
            label = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(parts[4])

            parsed_report[label] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support
            }

        elif len(parts) == 3 and parts[0].lower() == "accuracy":
            # Typically: ["accuracy", "0.71", "865"]
            # The middle one is the accuracy value
            # The last one is total support
            accuracy_value = float(parts[1])
            # If desired, you could store the support of accuracy as well
            # But typically we only keep the float for accuracy
            parsed_report["accuracy"] = accuracy_value

        else:
            # If there's any mismatch or unexpected line, handle or skip
            # print("Skipping line:", line)
            pass

    return parsed_report


if __name__ == "__main__":
    main()
