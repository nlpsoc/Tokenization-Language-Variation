"""
    calculate the correlations between BERT predictions and log regression / intrinsic measures
"""
import json
import os
import re
import statistics

from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
import seaborn as sns

import numpy as np
import pandas as pd

from styletokenizer.glue import GLUE_TEXTFLINT_TASKS, GLUE_MVALUE_TASKS, GLUE_TASKS
from styletokenizer.tokenizer import TOKENIZER_PATHS
from utility.datasets_helper import VARIETIES_TASKS

performance_keys = {
    "sst2": "eval_accuracy",
    "qqp": "eval_f1",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
    "NUCLE": "eval_f1",
    "PAN": "eval_accuracy",
}


def main():
    # do this only for the textflint tasks for now
    tasks = GLUE_TEXTFLINT_TASKS + GLUE_TASKS + GLUE_MVALUE_TASKS + VARIETIES_TASKS
    tokenizer_paths = TOKENIZER_PATHS

    unique_tokenizer_paths = set()
    for tokenizer_path in tokenizer_paths:
        for path in tokenizer_path:
            unique_tokenizer_paths.add(path)

    # collect the BERT performance scores of the tasks

    local_finder_addition = "/Users/anna/sftp_mount/hpc_disk/02-awegmann/"
    server_finder_addition = "/hpc/uu_cs_nlpsoc/02-awegmann/"

    BERT_PERFORMANCE = get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition,
                                             bert_version="base-BERT")  # train-mixed/base-BER
    df = pd.DataFrame(BERT_PERFORMANCE).T
    df.index.name = "BERT-Model"
    print(df.to_markdown())

    STATS_BASE_PATH = os.path.join(local_finder_addition, "TOKENIZER/tokenizer/")
    LOG_REGRESSION = get_logreg_performances(tasks, unique_tokenizer_paths, STATS_BASE_PATH)
    df = pd.DataFrame(LOG_REGRESSION).T
    df.index.name = "LR-Model"
    print(df.to_markdown())
    # calculate the correlation between the BERT performance and the logistic regression
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION)
    print(f"Correlation between BERT and LR: {c}")

    # get intrinisic measures
    intrinsic_key = "avg_seq_len"  # renyi_eff_2.5
    INTRINSIC_MEASURE = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    # calculate the correlation between the BERT performance and the intrinsic measures
    # TODO: do not caluclate correlation for the vocab size
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE)
    print(f"Correlation between BERT and seq len: {c}")

    intrinsic_key = "renyi_eff_2.5"
    INTRINSIC_MEASURE = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE)
    print(f"Correlation between BERT and renyi eff 2.5: {c}")

    # for all types of tasks: GLUE tasks, VARIETIES
    ROBUST_TASKS = GLUE_TEXTFLINT_TASKS  # + GLUE_TASKS
    # for grouped models (consider tokenizer paths groups)
    for g_nbr, tok_group in enumerate(TOKENIZER_PATHS):
        performance_per_tok = {}
        mean_performance_per_tok = {}
        for tokenizer_path in tok_group:
            tokenizer = os.path.basename(os.path.dirname(tokenizer_path))
            performance_per_tok[tokenizer] = {}
            for task in ROBUST_TASKS:
                if task in BERT_PERFORMANCE[tokenizer]:
                    performance_per_tok[tokenizer][task] = BERT_PERFORMANCE[tokenizer][task]
            # calculate mean performance robust task
            mean_performance_per_tok[tokenizer] = np.mean(list(performance_per_tok[tokenizer].values()))

        # calculate Wilcoxon signed-rank test
        # get models sorted by mean performance
        sorted_models = sorted(mean_performance_per_tok, key=mean_performance_per_tok.get, reverse=True)
        pval_matrix = pd.DataFrame(
            data=np.ones((len(sorted_models), len(sorted_models))),  # fill with 1.0 initially
            index=[m_name.split("-")[g_nbr] for m_name in sorted_models],
            columns=[m_name.split("-")[g_nbr] for m_name in sorted_models],
        )
        for i in range(len(sorted_models)):
            for j in range(i + 1, len(sorted_models)):
                data1 = []
                data2 = []
                tok1 = sorted_models[i]
                tok2 = sorted_models[j]
                for task in ROBUST_TASKS:
                    if task in performance_per_tok[tok1] and task in performance_per_tok[tok2]:
                        data1.append(performance_per_tok[tok1][task])
                        data2.append(performance_per_tok[tok2][task])
                if len(data1) > 0:
                    stat, pval = wilcoxon(data1, data2, alternative="greater")
                    pval_matrix.iloc[i, j] = pval
                    pval_matrix.iloc[j, i] = pval
                else:
                    pval_matrix.iloc[i, j] = None
                    pval_matrix.iloc[j, i] = None

        # pval_matrix = bonferroni_correction(pval_matrix, sorted_models)

        print("Pairwise Wilcoxon p-value matrix (Bonferroni-corrected):")
        print(pval_matrix)

        mask = np.triu(np.ones_like(pval_matrix, dtype=bool))  # hides upper triangle + diagonal

        plt.figure(figsize=(6, 5))
        sns.heatmap(pval_matrix,
                    mask=mask,
                    annot=True,  # show p-values in each cell
                    cmap="coolwarm_r",  # color scheme
                    fmt=".3f",  # formatting for p-values
                    vmin=0, vmax=1)

        plt.title("Pairwise Wilcoxon p-values (Heatmap)")
        plt.show()


def bonferroni_correction(pval_matrix, sorted_models):
    all_pvals = []
    for x in range(len(sorted_models)):
        for y in range(x + 1, len(sorted_models)):
            all_pvals.append(pval_matrix.iloc[x, y])
    # Apply Bonferroni
    from math import ceil
    adjusted = [min(p * len(all_pvals), 1.0) for p in all_pvals]
    # Put them back into the matrix
    idx = 0
    for x in range(len(sorted_models)):
        for y in range(x + 1, len(sorted_models)):
            pval_matrix.iloc[x, y] = adjusted[idx]
            pval_matrix.iloc[y, x] = adjusted[idx]
            idx += 1
    return pval_matrix


def get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH):
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
        if task in GLUE_TEXTFLINT_TASKS or task in GLUE_MVALUE_TASKS:
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


def get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition, bert_version="base-BERT"):
    BERT_PERFORMANCE = {}
    base_out_base_path = os.path.join(local_finder_addition, "TOKENIZER/output/")
    BERT_PATH = "749M/steps-45000/seed-42/42/"
    for task in tasks:
        if task in GLUE_TEXTFLINT_TASKS:
            task_finder_addition = f"GLUE/textflint/{bert_version}/"
            results_out_base_path = os.path.join(base_out_base_path, task_finder_addition)
        elif task in GLUE_MVALUE_TASKS:
            task_finder_addition = f"GLUE/mVALUE/{bert_version}/"
            results_out_base_path = os.path.join(base_out_base_path, task_finder_addition)
        elif task in GLUE_TASKS:
            task_finder_addition = f"GLUE/{bert_version}/"
            results_out_base_path = os.path.join(base_out_base_path, task_finder_addition)
        elif task in VARIETIES_TASKS:
            task_finder_addition = f"VAR/{bert_version}/"
            results_out_base_path = os.path.join(base_out_base_path, task_finder_addition)
        else:
            raise NotImplementedError("Only textflint tasks are implemented")
        task_key = task
        if task in GLUE_TEXTFLINT_TASKS or task in GLUE_MVALUE_TASKS:
            task_key = task.split('-')[0]

        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            if tokenizer_name not in BERT_PERFORMANCE:
                BERT_PERFORMANCE[tokenizer_name] = {}
            # get the BERT output for the task
            result_path = os.path.join(results_out_base_path, tokenizer_name, BERT_PATH, task_key, "all_results.json")
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
