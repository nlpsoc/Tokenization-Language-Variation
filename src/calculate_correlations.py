"""
    calculate the correlations between BERT predictions and log regression / intrinsic measures
"""
import json
import os
import re
import statistics
from statsmodels.stats.contingency_tables import mcnemar
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
import seaborn as sns
import statsmodels.formula.api as smf

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
    "CORE": "eval_accuracy",
    "multi-DIALECT": "eval_accuracy",
    "sadiri": "Test accuracy",
}


def save_dict_as_json(data, file_path):
    """
    Converts a dictionary containing non-serializable objects (e.g., numpy arrays)
    to a JSON-serializable format and saves it to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the JSON file.
    """

    def make_serializable(obj):
        # Recursively convert numpy arrays to lists
        if isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert the dictionary to a serializable format
    serializable_data = make_serializable(data)

    # check if folder exists
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save to a JSON file
    with open(file_path, "w") as file:
        json.dump(serializable_data, file, indent=4)


def load_json_as_dict(file_path):
    """
    Loads a JSON file and converts lists back to numpy arrays where applicable.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded dictionary with lists converted to numpy arrays.
    """

    def convert_to_numpy(obj):
        # Recursively convert lists back to numpy arrays
        if isinstance(obj, dict):
            return {key: convert_to_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            # Check if it should be converted to a numpy array
            try:
                return np.array(obj)
            except:
                return [convert_to_numpy(item) for item in obj]
        else:
            return obj

    # Load the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Convert lists back to numpy arrays
    return convert_to_numpy(data)


def main():
    # do this only for the textflint tasks for now
    tasks = GLUE_TEXTFLINT_TASKS + GLUE_TASKS + GLUE_MVALUE_TASKS + VARIETIES_TASKS
    tokenizer_paths = TOKENIZER_PATHS

    unique_tokenizer_paths = set()
    for tokenizer_path in tokenizer_paths:
        for path in tokenizer_path:
            unique_tokenizer_paths.add(path)

    # collect the BERT performance scores of the tasks

    local_finder_addition = "/Users/anna/sftp_mount/hpc_disk6/02-awegmann/"
    server_finder_addition = "/hpc/uu_cs_nlpsoc/02-awegmann/"

    bert_version = "train-mixed/base-BERT"  # train-mixed/base-BER
    BERT_PERFORMANCE = get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition,
                                             bert_version=bert_version)
    if os.path.exists(f"{bert_version}_predictions.json"):
        BERT_PREDICTIONS = load_json_as_dict(f"{bert_version}_predictions.json")
    else:
        BERT_PREDICTIONS = get_BERT_predictions(tasks, unique_tokenizer_paths, local_finder_addition,
                                                bert_version=bert_version)  # train-mixed/base-BERT
        save_dict_as_json(BERT_PREDICTIONS, f"{bert_version}_predictions.json")
    df = pd.DataFrame(BERT_PERFORMANCE).T
    df.index.name = "BERT-Model"
    # reorder df in the order "twitter-gpt2-32000", "mixed-
    print(df.to_markdown())
    calculate_robustness_scores(BERT_PERFORMANCE, model_name="BERT-Model")

    STATS_BASE_PATH = os.path.join(local_finder_addition, "TOKENIZER/tokenizer/")
    LOG_REGRESSION = get_logreg_performances(tasks, unique_tokenizer_paths, STATS_BASE_PATH)
    df = pd.DataFrame(LOG_REGRESSION).T
    df.index.name = "LR-Model"
    print(df.to_markdown())
    # calculate the correlation between the BERT performance and the logistic regression
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION)
    print(f"Correlation between BERT and LR: {c}")
    c_no_size = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, no_size_difference=True)
    print(f"Correlation between BERT and LR (no size difference): {c_no_size}")
    calculate_robustness_scores(LOG_REGRESSION, model_name="LR")

    # get intrinisic measures
    intrinsic_key = "avg_seq_len"  # renyi_eff_2.5
    INTRINSIC_MEASURE = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    # calculate the correlation between the BERT performance and the intrinsic measures
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE, no_size_difference=True)
    print(f"Correlation between BERT and seq len: {c}")

    intrinsic_key = "renyi_eff_2.5"
    INTRINSIC_MEASURE = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    print(INTRINSIC_MEASURE)
    c = calculate_correlation(BERT_PERFORMANCE, INTRINSIC_MEASURE, no_size_difference=True)
    print(f"Correlation between BERT and renyi eff 2.5: {c}")

    # for all types of tasks: GLUE tasks, VARIETIES
    ROBUST_TASKS = GLUE_TEXTFLINT_TASKS + GLUE_TASKS + GLUE_MVALUE_TASKS
    significance_test(ROBUST_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, tasks_name="robust")
    significance_test(VARIETIES_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, tasks_name="sensitive")
    mixed_effect_model(BERT_PERFORMANCE, ROBUST_TASKS)
    mixed_effect_model(BERT_PERFORMANCE, VARIETIES_TASKS)

    # Wilcoxon test for the BERT predictions
    significance_test(ROBUST_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, do_wilcoxon=True, tasks_name="robust")
    significance_test(VARIETIES_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, do_wilcoxon=True, tasks_name="sensitive")


def significance_test(considered_tasks, performance_values, predictions_for_mcnemar=None, do_wilcoxon=False,
                      bonferroni=True, tasks_name="robust"):
    # for grouped models (consider tokenizer paths groups)
    for g_nbr, tok_group in enumerate(TOKENIZER_PATHS):
        performance_per_tok = {}
        mean_performance_per_tok = {}
        for tokenizer_path in tok_group:
            tokenizer = os.path.basename(os.path.dirname(tokenizer_path))
            performance_per_tok[tokenizer] = {}
            for task in considered_tasks:
                if task in performance_values[tokenizer]:
                    performance_per_tok[tokenizer][task] = performance_values[tokenizer][task]

            # calculate mean performance robust task
            mean_performance_per_tok[tokenizer] = np.mean(list(performance_per_tok[tokenizer].values()))

        # get models sorted by mean performance
        sorted_models_names = sorted(mean_performance_per_tok, key=mean_performance_per_tok.get, reverse=True)
        pval_matrix = pd.DataFrame(
            data=np.ones((len(sorted_models_names), len(sorted_models_names))),  # fill with 1.0 initially
            index=[m_name.split("-")[g_nbr] for m_name in sorted_models_names],
            columns=[m_name.split("-")[g_nbr] for m_name in sorted_models_names],
        )

        for i in range(len(sorted_models_names)):
            for j in range(i + 1, len(sorted_models_names)):
                data1 = []
                data2 = []
                tok1 = sorted_models_names[i]
                tok2 = sorted_models_names[j]
                correct = []
                for task in considered_tasks:
                    if task in performance_per_tok[tok1] and task in performance_per_tok[tok2]:
                        if do_wilcoxon:
                            data1.append(performance_per_tok[tok1][task])
                            data2.append(performance_per_tok[tok2][task])
                        else:
                            if task not in predictions_for_mcnemar[tok1] or task not in predictions_for_mcnemar[tok2]:
                                print(f"ERROR: Predictions {task} not found for {tok1} or {tok2}")
                                continue
                            data1 += list(predictions_for_mcnemar[tok1][task])
                            data2 += list(predictions_for_mcnemar[tok2][task])
                            correct += list(predictions_for_mcnemar["gt"][task])
                if len(data1) > 0:
                    if do_wilcoxon:
                        stat, pval = wilcoxon(data1, data2, alternative="greater")
                        pval_matrix.iloc[i, j] = pval
                        pval_matrix.iloc[j, i] = pval
                    else:
                        # flatten arrays
                        stat, pval, table = compute_mcnemar_statsmodels(correct, data1, data2)
                        pval_matrix.iloc[i, j] = pval
                        pval_matrix.iloc[j, i] = pval
                else:
                    pval_matrix.iloc[i, j] = None
                    pval_matrix.iloc[j, i] = None

        addition = ""
        if bonferroni:
            pval_matrix = bonferroni_correction(pval_matrix, sorted_models_names)
            addition = "Bonferroni-corrected"

        if do_wilcoxon:
            print(f"Pairwise Wilcoxon p-value matrix {addition}:")
        else:
            print(f"Pairwise McNemar p-value matrix {addition}:")
        print(pval_matrix)

        mask = np.triu(np.ones_like(pval_matrix, dtype=bool))  # hides upper triangle + diagonal
        sns.set_context("paper")

        plt.figure(figsize=(6, 5))
        sns.heatmap(pval_matrix,
                    mask=mask,
                    annot=True,  # show p-values in each cell
                    cmap="coolwarm",  # color scheme
                    fmt=".3f",  # formatting for p-values
                    vmin=0, vmax=1)

        test = "Wilcoxon" if do_wilcoxon else "McNemar"
        plt.title(f"Pairwise {test} p-values (Heatmap) on {tasks_name} Tasks")
        plt.show()


def mixed_effect_model(performance_per_tok, tasks=GLUE_TASKS):
    data_list = []
    for top_key, task_dict in performance_per_tok.items():  # performance_values

        # Each top-level key looks like "mixed-gpt2-32000"
        # Split it by '-' to extract Domain, Model, and VocabSize.
        domain, base_model, vocab_str = top_key.split('-')

        for task_name, performance in task_dict.items():
            if task_name not in tasks:
                continue
            data_list.append({
                "Corpus": domain,
                "Model": base_model,
                "VocabSize": vocab_str,  # could convert to int if you like: int(vocab_str)
                "Task": task_name,  # e.g. "sst2-textflint"
                "Performance": performance
            })
    # 2. Create a DataFrame
    df = pd.DataFrame(data_list)
    # 3. Fit a mixed-effects model.
    #    Example: treat Domain, Model, VocabSize as fixed effects
    #    and allow random intercepts by Task (since many tasks repeat).
    model = smf.mixedlm(
        formula="Performance ~ C(Corpus) + C(Model) + C(VocabSize) + C(Task)",
        data=df,
        groups=df["Task"]  # random grouping by Task
    )
    result = model.fit()
    print(result.summary())


def calculate_robustness_scores(model_result_dict, model_name="LR-Model"):
    # calculate Mean and STD for LR, for ROBUSTNESS
    mean_std_dict = {}
    for tokenizer_name in model_result_dict.keys():
        mean_std_dict[tokenizer_name] = {}
        if all([key in model_result_dict[tokenizer_name] for key in GLUE_TASKS]):
            mean_glue = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_TASKS])
            # std = np.std([LOG_REGRESSION[tokenizer_name][key] for key in GLUE_TASKS])
            mean_std_dict[tokenizer_name]["GLUE"] = {"mean": mean_glue}  # , "std": std
            if all([key in model_result_dict[tokenizer_name] for key in GLUE_TEXTFLINT_TASKS]):
                mean = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_TEXTFLINT_TASKS])
                # std = np.std([LOG_REGRESSION[tokenizer_name][key] for key in GLUE_TEXTFLINT_TASKS])
                mean_std_dict[tokenizer_name]["GLUE_TEXTFLINT"] = {"mean": mean,
                                                                   "reduction": mean - mean_glue}  # , "std": std
            # check if the tokenizer has all mVALUE tasks
            if all([key in model_result_dict[tokenizer_name] for key in GLUE_MVALUE_TASKS]):
                mean = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_MVALUE_TASKS])
                # std = np.std([LOG_REGRESSION[tokenizer_name][key] for key in GLUE_MVALUE_TASKS])
                mean_std_dict[tokenizer_name]["GLUE_MVALUE"] = {"mean": mean,
                                                                "reduction": mean - mean_glue}  # , "std": std
    df = pd.DataFrame(mean_std_dict).T
    df.index.name = model_name
    print(df.to_markdown())


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


def calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, no_size_difference=False):
    x = []
    y = []
    for tokenizer_name in BERT_PERFORMANCE:
        size = int(tokenizer_name.split("-")[-1])
        if no_size_difference and size != 32000:
            continue
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
        report_file = "classification_report.txt"
        if task in GLUE_TEXTFLINT_TASKS or task in GLUE_MVALUE_TASKS:
            task_key = task.split('-')[0]
        if (task == "NUCLE") or (task == "CORE"):
            report_file = "f1_per_label.txt"
        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            if tokenizer_name not in LOG_REGRESSION:
                LOG_REGRESSION[tokenizer_name] = {}
            # get the BERT output for the task
            result_path = os.path.join(stats_base_path, tokenizer_name, LR_ADDITION, task, report_file)

            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue
            classification_report = parse_classification_report(result_path)

            # get the performance from the performance keys
            if "accuracy" in performance_keys[task_key]:
                if "accuracy" not in classification_report:
                    print(f"Accuracy not found in {result_path}")
                    continue
                LOG_REGRESSION[tokenizer_name][task] = classification_report["accuracy"]
            else:
                if task == "NUCLE":
                    LOG_REGRESSION[tokenizer_name][task] = classification_report["f1_macro"]
                else:
                    LOG_REGRESSION[tokenizer_name][task] = classification_report['1']["f1-score"]
    return LOG_REGRESSION


def get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition, bert_version="base-BERT"):
    BERT_PERFORMANCE = {}
    base_out_base_path = os.path.join(local_finder_addition, "TOKENIZER/output/")
    BERT_PATH = "749M/steps-45000/seed-42/42/"

    for task in tasks:
        result_file = "all_results.json"
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
            if task == "sadiri":
                result_file = "test_accuracy.txt"
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
            result_path = os.path.join(results_out_base_path, tokenizer_name, BERT_PATH, task_key, result_file)
            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue

            if not task == "sadiri":
                # read in json file
                with open(result_path, "r") as f:
                    data_dict = json.load(f)
            else:
                data_dict = parse_sadiri_metrics(result_path)
            if performance_keys[task_key] in data_dict:
                # get the performance from the performance keys
                BERT_PERFORMANCE[tokenizer_name][task] = data_dict[performance_keys[task_key]]
            else:
                print(f"Performance key {performance_keys[task_key]} not found in {result_path}")
                continue
    return BERT_PERFORMANCE


def get_BERT_predictions(tasks, unique_tokenizer_paths, local_finder_addition, bert_version="base-BERT"):
    BERT_PREDICTIONS = {}
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
            raise NotImplementedError(f"task {task} not found")

        task_key = task
        if task in GLUE_TEXTFLINT_TASKS or task in GLUE_MVALUE_TASKS:
            task_key = task.split('-')[0]

        for tokenizer_path in unique_tokenizer_paths:
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            result_path = os.path.join(results_out_base_path, tokenizer_name, BERT_PATH, task_key)
            # check if there exists a tsv file including "eval_dataset" in the name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))

            # get the BERT output for the task
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue
            files = os.listdir(result_path)
            for file in files:
                if ((task_key in GLUE_TASKS and f"eval_dataset_{task_key}.tsv" in file) or
                        (task_key not in GLUE_TASKS and "eval_dataset.tsv" in file)):
                    result_path = os.path.join(result_path, file)
                    # read in .tsv file
                    df = pd.read_csv(result_path, sep='\t')
                    # save the predictions from the "predictions" column
                    if tokenizer_name not in BERT_PREDICTIONS:
                        BERT_PREDICTIONS[tokenizer_name] = {}
                    if "gt" not in BERT_PREDICTIONS:
                        BERT_PREDICTIONS["gt"] = {}
                    if task not in BERT_PREDICTIONS["gt"]:
                        BERT_PREDICTIONS["gt"][task] = df["label"].values
                    BERT_PREDICTIONS[tokenizer_name][task] = df["predictions"].values
                    break

            if (tokenizer_name not in BERT_PREDICTIONS) or (task not in BERT_PREDICTIONS[tokenizer_name]):
                print(f"No predictions found for {tokenizer_name} ...")
                continue

    return BERT_PREDICTIONS


def parse_classification_report(file_path):
    """
    Parse either:
      1) A standard sklearn.metrics.classification_report text output
         into a structured dictionary with per-class metrics, plus accuracy.
      2) A file containing lines like:
         F1 per label: [0.14567527 0.02362205 ... 0.50424929]
         F1 weighted: 0.1291048390401571
         F1 macro: 0.0832130497702279

    Returns a dictionary, for example:
    {
       "0": {"precision": 0.71, "recall": 0.67, "f1-score": 0.69, "support": 422},
       "1": {...},
       "accuracy": 0.71,
       "macro avg": {...},
       "weighted avg": {...},
       "f1_per_label": [0.14567527, 0.02362205, 0.15300546, ...],
       "f1_weighted": 0.1291048390401571,
       "f1_macro": 0.0832130497702279
    }
    """

    parsed_report = {}

    if "NUCLE" in file_path:
        print("DEBUG")

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Regex to split the standard classification report lines on 2+ spaces:
    splitter = re.compile(r"\s{2,}")

    # A small helper to parse lines that look like "F1 weighted: 0.12345"
    def parse_f1_line(line, prefix):
        # e.g., line == "F1 weighted: 0.1291048390401571"
        # prefix might be "F1 weighted:"
        # We return the float part after the prefix
        # We'll assume well-formed input
        return float(line.replace(prefix, "").strip())

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # 1) Detect F1 per label bracketed array
        if line.startswith("F1 per label:"):
            # We may need to accumulate multiple lines until we reach a ']'
            f1_block = line
            i += 1
            # If the line doesn't contain the closing bracket, keep accumulating
            while i < n and ']' not in f1_block:
                f1_block += ' ' + lines[i]
                i += 1
            i -= 1  # rewind by one to be on closing bracket line
            # Now parse all floats inside the brackets
            # e.g. f1_block might be: "F1 per label: [0.14 0.02 0.15 ... 0.50]"
            # Let's extract the portion inside [  ...  ]
            match = re.search(r'\[([^\]]+)\]', f1_block)
            if match:
                float_str = match.group(1)  # everything inside the brackets
                # split on whitespace to get each potential float
                str_vals = float_str.split()
                # Convert each to float
                values = []
                for val in str_vals:
                    # handle possible "0." or numeric strings
                    # Python can handle "0." but let's be robust if there's a trailing '.'
                    # or if they are "0.000" etc
                    # We'll just do float(val).
                    values.append(float(val))

                parsed_report["f1_per_label"] = values

        # 2) Detect "F1 weighted: x.xxxxxx"
        elif line.startswith("F1 weighted:"):
            parsed_report["f1_weighted"] = parse_f1_line(line, "F1 weighted:")

        # 3) Detect "F1 macro: x.xxxxxx"
        elif line.startswith("F1 macro:"):
            parsed_report["f1_macro"] = parse_f1_line(line, "F1 macro:")

        # 4) Detect "F1 micro: x.xxxxxx"
        elif line.startswith("F1 micro:"):
            parsed_report["f1_micro"] = parse_f1_line(line, "F1 micro:")

        # 5) Detect "Accuracy: x.xxxxxx"
        elif line.startswith("Accuracy:"):
            parsed_report["accuracy"] = parse_f1_line(line, "Accuracy:")

        else:
            # 4) Try to parse as a standard classification report line
            parts = splitter.split(line)

            if len(parts) == 5:
                # Likely a class label or an average line
                label = parts[0]
                try:
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
                except ValueError:
                    # If it doesn't parse correctly, skip or handle differently
                    pass

            elif len(parts) == 3 and parts[0].lower() == "accuracy":
                # Typically: ["accuracy", "0.71", "865"]
                try:
                    accuracy_value = float(parts[1])
                    parsed_report["accuracy"] = accuracy_value
                except ValueError:
                    pass
            # else: it's a line we don't parse or don't recognize

            i += 1
            continue

        i += 1

    return parsed_report


def parse_sadiri_metrics(filepath):
    """
    Reads a text file with lines in the form:
        <key>: <value>
    and returns a dictionary with parsed values (floats where possible).
    """
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Split only on the first ':' to allow keys with extra ':' characters
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue  # If the format doesn't match 'key: value', skip
            key, val_str = parts
            key = key.strip()
            val_str = val_str.strip()

            # Try to convert the value to float
            try:
                value = float(val_str)
            except ValueError:
                # If it can't be converted to float, leave it as string
                value = val_str

            results[key] = value
    return results


def compute_mcnemar_statsmodels(y_true, y_pred1, y_pred2, exact=False, correction=True):
    """
    Compute McNemar's test using statsmodels for two classification models
    given the ground truth and their predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred1 : array-like of shape (n_samples,)
        Predictions from model A.
    y_pred2 : array-like of shape (n_samples,)
        Predictions from model B.
    exact : bool, optional
        Whether to use the exact binomial test. Default is False
        (uses the chi-square approximation).
    correction : bool, optional
        Whether to apply continuity correction when using the chi-square
        approximation. Default is True.

    Returns
    -------
    statistic : float
        Test statistic (chi-square statistic if exact=False, or the number
        of discordant pairs if exact=True).
    pvalue : float
        Two-sided p-value for the test.
    table : 2D numpy array
        The 2×2 contingency table:
            [[a, b],
             [c, d]]
    """
    # if type(y_true[0]) == list:
    #     y_true = [item for sublist in y_true for item in sublist]
    #     y_pred1 = [item for sublist in y_pred1 for item in sublist]
    #     y_pred2 = [item for sublist in y_pred2 for item in sublist]
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    # Ensure all arrays have the same length
    if not (len(y_true) == len(y_pred1) == len(y_pred2)):
        raise ValueError("All input arrays must have the same length.")

    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    a = np.sum(correct1 & correct2)  # both correct
    b = np.sum(correct1 & ~correct2)  # model A correct, model B wrong
    c = np.sum(~correct1 & correct2)  # model A wrong, model B correct
    d = np.sum(~correct1 & ~correct2)  # both wrong

    # Build the 2×2 table
    table = np.array([[a, b],
                      [c, d]])

    # Perform McNemar's test using statsmodels
    result = mcnemar(table, exact=exact, correction=correction)

    # The result object has two main attributes: statistic and pvalue
    return result.statistic, result.pvalue, table


if __name__ == "__main__":
    main()
