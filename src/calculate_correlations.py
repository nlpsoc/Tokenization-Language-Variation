"""
    dirty script to collect the experiment results
     i.e., the performance scores of BERT, log regression and the intrinsic measures,
        also calculates the correlations
    Note: this expects a certain path structure and file structure with results
"""
import json
import math
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

from styletokenizer.robust_tasks import GLUE_TEXTFLINT_TASKS, GLUE_MVALUE_TASKS, GLUE_TASKS
from utility.tokenizer_vars import TOKENIZER_PATHS
from sensitive_tasks import SENSITIVE_TASKS

SEEDS = [42, 43, 44]
ROBUST_TASKS = GLUE_TEXTFLINT_TASKS + GLUE_TASKS + GLUE_MVALUE_TASKS

INDENTIFIABLE_PATH = "/Users/anna/sftp_mount/hpc_disk/02-awegmann/"

AV_TASK = "sadiri"
performance_keys = {
    "sst2": "eval_accuracy",
    "qqp": "eval_f1",
    "mnli": ["eval_accuracy", "eval_accuracy_mm"],
    "qnli": "eval_accuracy",
    "ANLI": "eval_accuracy",
    "NUCLE": "eval_f1",
    "PAN": "eval_accuracy",
    "CORE": "eval_accuracy",
    "multi-DIALECT": "eval_accuracy",
    AV_TASK: "Test accuracy",
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
    if folder != "" and not os.path.exists(folder):
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
    tasks = GLUE_TEXTFLINT_TASKS + GLUE_TASKS + GLUE_MVALUE_TASKS + SENSITIVE_TASKS
    tokenizer_paths = TOKENIZER_PATHS

    unique_tokenizer_paths = set()
    for tokenizer_path in tokenizer_paths:
        for path in tokenizer_path:
            unique_tokenizer_paths.add(path)

    # collect the BERT performance scores of the tasks

    local_base_path = INDENTIFIABLE_PATH

    # check if local finder addition exists
    if not os.path.exists(local_base_path):
        raise FileNotFoundError(f"Local finder addition {local_base_path} does not exist")

    bert_version = "train-mixed/base-BERT"
    if "train" in bert_version:
        # remove "mixed" from first tokenizer group
        TOKENIZER_PATHS[0] = [path for path in TOKENIZER_PATHS[0] if "mixed" not in path]
    if os.path.exists(f"_tmp/{bert_version}_performances.json"):
        BERT_PERFORMANCE = load_json_as_dict(f"_tmp/{bert_version}_performances.json")
    else:
        BERT_PERFORMANCE = get_BERT_performances(tasks, unique_tokenizer_paths, local_base_path,
                                                 bert_version=bert_version)
        save_dict_as_json(BERT_PERFORMANCE, f"_tmp/{bert_version}_performances.json")
    if os.path.exists(f"_tmp/{bert_version}_predictions.json"):
        BERT_PREDICTIONS = load_json_as_dict(f"_tmp/{bert_version}_predictions.json")
    else:
        BERT_PREDICTIONS = get_BERT_predictions(tasks, unique_tokenizer_paths, local_base_path,
                                                bert_version=bert_version)  # train-mixed/base-BERT
        save_dict_as_json(BERT_PREDICTIONS, f"_tmp/{bert_version}_predictions.json")
    # df = pd.DataFrame(BERT_PERFORMANCE).T

    # add a mean-robust and a mean-sensitive column
    BERT_performance_df = pd.DataFrame({
        model: {
            # Mean and std of means across robust tasks per seed
            "mean-robust": np.mean(seed_means := [
                np.mean([seed_dict[task] for task in ROBUST_TASKS if task in seed_dict])
                for seed, seed_dict in model_dict.items() if int(seed) in SEEDS
            ]),
            "std-robust": np.std(seed_means),
            # Individual task means across seeds
            "mean-sensitive": np.mean(sensitive_means := [
                np.mean([seed_dict[task] for task in SENSITIVE_TASKS if task in seed_dict])
                for seed, seed_dict in model_dict.items() if int(seed) in SEEDS
            ]),
            "std-sensitive": np.std(sensitive_means),
            **{
                f"{task}-mean": np.mean([
                    seed_dict[task] for seed, seed_dict in model_dict.items()
                    if int(seed) in SEEDS and task in seed_dict
                ])
                for task in ROBUST_TASKS
            },
            **{
                f"{task}-std": np.std([
                    seed_dict[task] for seed, seed_dict in model_dict.items()
                    if int(seed) in SEEDS and task in seed_dict
                ])
                for task in ROBUST_TASKS
            },
            **{
                f"{task}-mean": np.mean([
                    seed_dict[task] for seed, seed_dict in model_dict.items()
                    if int(seed) in SEEDS and task in seed_dict
                ])
                for task in SENSITIVE_TASKS
            },
            **{
                f"{task}-std": np.std([
                    seed_dict[task] for seed, seed_dict in model_dict.items()
                    if int(seed) in SEEDS and task in seed_dict
                ])
                for task in SENSITIVE_TASKS
            },
        }
        for model, model_dict in BERT_PERFORMANCE.items()
    }).T
    BERT_performance_df.index.name = "BERT-Model"

    # df["mean-robust"] = df[SEEDS, ROBUST_TASKS].mean(axis=1)
    # df["mean-sensitive"] =  df[VARIETIES_TASKS].mean(axis=1)

    # reorder df in the order "twitter-gpt2-32000", "mixed-
    print(BERT_performance_df.to_markdown())
    calculate_robustness_scores(BERT_PERFORMANCE, model_name="BERT-Model")

    STATS_BASE_PATH = os.path.join(local_base_path, "TOKENIZER/tokenizer/")
    if os.path.exists(f"_tmp/LR/{bert_version}_performances.json"):
        LOG_REGRESSION = load_json_as_dict(f"_tmp/LR/{bert_version}_performances.json")
    else:
        LOG_REGRESSION = get_logreg_performances(tasks, unique_tokenizer_paths, STATS_BASE_PATH)
        save_dict_as_json(LOG_REGRESSION, f"_tmp/LR/{bert_version}_performances.json")
    df = pd.DataFrame(LOG_REGRESSION).T
    df.index.name = "LR-Model"
    df["mean-robust"] = df[ROBUST_TASKS].mean(axis=1)
    df["mean-sensitive"] = df[SENSITIVE_TASKS].mean(axis=1)
    # print(df.to_markdown())
    # calculate the correlation between the BERT performance and the logistic regression
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION)
    print(f"Correlation between BERT and LR: {c}")
    c_no_size = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, no_size_difference=True)
    print(f"Correlation between BERT and LR (no size difference): {c_no_size}")
    calculate_robustness_scores(LOG_REGRESSION, model_name="LR")

    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, only_robust=True)
    print(f"Correlation between BERT and LR (robust tasks): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, no_size_difference=True, only_robust=True)
    print(f"Correlation between BERT and LR (robust tasks, no size difference): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, only_sensitive=True)
    print(f"Correlation between BERT and LR (sensitive tasks): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, LOG_REGRESSION, no_size_difference=True, only_sensitive=True)
    print(f"Correlation between BERT and LR (sensitive tasks, no size difference): {c}")

    # get intrinisic measures
    intrinsic_key = "avg_seq_len"  # renyi_eff_2.5
    seq_len = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    df = pd.DataFrame(seq_len).T
    df.index.name = "Seq Len"
    # print(df.to_markdown())
    # calculate the correlation between the BERT performance and the intrinsic measures
    c = calculate_correlation(BERT_PERFORMANCE, seq_len, no_size_difference=True)
    print(f"Correlation between BERT and seq len: {c}")
    c = calculate_correlation(BERT_PERFORMANCE, seq_len, no_size_difference=True, only_robust=True)
    print(f"Correlation between BERT and seq len (robust tasks): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, seq_len, no_size_difference=True, only_sensitive=True)
    print(f"Correlation between BERT and seq len (sensitive tasks): {c}")

    intrinsic_key = "renyi_eff_2.5"
    renyi = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    df = pd.DataFrame(renyi).T
    df.index.name = "Renyi Eff 2.5"
    # print(df.to_markdown())
    c = calculate_correlation(BERT_PERFORMANCE, renyi, no_size_difference=True)
    print(f"Correlation between BERT and renyi eff 2.5: {c}")
    # correlation Renyi and BERT on robust tasks
    c = calculate_correlation(BERT_PERFORMANCE, renyi, no_size_difference=True, only_robust=True)
    print(f"Correlation between BERT and renyi eff 2.5 (robust tasks): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, renyi, no_size_difference=True, only_sensitive=True)
    print(f"Correlation between BERT and renyi eff 2.5 (sensitive tasks): {c}")

    intrinsic_key = "vocab_size"
    vocab_size = get_intrinsic_performances(tasks, unique_tokenizer_paths, intrinsic_key, STATS_BASE_PATH)
    df = pd.DataFrame(vocab_size).T
    df.index.name = "Vocab Size"
    # print(df.to_markdown())
    c = calculate_correlation(BERT_PERFORMANCE, vocab_size, no_size_difference=True)
    print(f"Correlation between BERT and vocab size: {c}")
    c = calculate_correlation(renyi, vocab_size, no_size_difference=True, no_corpus_difference=True)
    print(f"Correlation between renyi and vocab size (only on pre-tokenizer): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, vocab_size, no_size_difference=True, only_robust=True)
    print(f"Correlation between BERT and vocab size (robust tasks): {c}")
    c = calculate_correlation(BERT_PERFORMANCE, vocab_size, no_size_difference=True, only_sensitive=True)
    print(f"Correlation between BERT and vocab size (sensitive tasks): {c}")

    # for all types of tasks: GLUE tasks, VARIETIES

    significance_test(ROBUST_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, tasks_name="robust")
    significance_test(SENSITIVE_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, tasks_name="sensitive")
    # mixed_effect_model(BERT_PERFORMANCE, ROBUST_TASKS)
    # mixed_effect_model(BERT_PERFORMANCE, SENSITIVE_TASKS)

    # Wilcoxon test for the BERT predictions
    # significance_test(ROBUST_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, do_wilcoxon=True, tasks_name="robust")
    # significance_test(SENSITIVE_TASKS, BERT_PERFORMANCE, BERT_PREDICTIONS, do_wilcoxon=True, tasks_name="sensitive")


def significance_test(considered_tasks, performance_values, predictions_for_mcnemar=None, do_wilcoxon=False,
                      bonferroni=True, tasks_name="robust"):
    def is_seed_nested(d):
        return isinstance(d, dict) and all(str(k).isdigit() for k in d)

    def get_mean_performance(perf, task):
        """Average performance across seeds or return directly."""
        if is_seed_nested(perf):
            vals = [perf[seed][task] for seed in perf if task in perf[seed] and not np.isnan(perf[seed][task])]
            return np.mean(vals) if vals else None
        return perf.get(task) if task in perf and not np.isnan(perf[task]) else None

    def get_concat_predictions(preds, task):
        """Concatenate predictions across seeds or return directly."""
        if is_seed_nested(preds):
            return [p for seed in preds if task in preds[seed] for p in preds[seed][task]]
        return list(preds[task]) if task in preds else None

    for g_nbr, tok_group in enumerate(TOKENIZER_PATHS):
        show_legend = g_nbr == len(TOKENIZER_PATHS) - 1
        performance_per_tok = {}
        mean_performance_per_tok = {}

        # Step 1: Collect and average performance per tokenizer
        for tokenizer_path in tok_group:
            tokenizer = os.path.basename(os.path.dirname(tokenizer_path))
            tok_perf = performance_values[tokenizer]
            task_scores = {
                task: get_mean_performance(tok_perf, task)
                for task in considered_tasks
            }
            task_scores = {task: val for task, val in task_scores.items() if val is not None}
            performance_per_tok[tokenizer] = task_scores
            mean_performance_per_tok[tokenizer] = np.mean(list(task_scores.values()))

        # Step 2: Sort tokenizers and prepare display names
        sorted_models = sorted(mean_performance_per_tok, key=mean_performance_per_tok.get, reverse=True)
        display_translations = {
            "twitter": "X", "mixed": "Mixed", "wikipedia": "Wiki", "pubmed": "PMed",
            "gpt2": "gpt", "32000": "32", "64000": "64", "128000": "128", "4000": "4",
            "500": "0.5", "llama3": "llama", "ws": "_ws", "wsorg": "ws", "no": "no",
        }

        def display_name(tok):
            return display_translations.get(tok.split("-")[g_nbr], tok.split("-")[g_nbr])

        pval_matrix = pd.DataFrame(
            np.ones((len(sorted_models), len(sorted_models))),
            index=[display_name(m) for m in sorted_models],
            columns=[display_name(m) for m in sorted_models],
        )

        # Step 3: Pairwise statistical tests
        for i, tok1 in enumerate(sorted_models):
            for j, tok2 in enumerate(sorted_models[i + 1:], start=i + 1):
                data1, data2, correct = [], [], []

                for task in considered_tasks:
                    if task not in performance_per_tok[tok1] or task not in performance_per_tok[tok2]:
                        continue

                    if do_wilcoxon:
                        data1.append(performance_per_tok[tok1][task])
                        data2.append(performance_per_tok[tok2][task])
                    else:
                        # McNemar: Gather predictions
                        preds1 = predictions_for_mcnemar.get(tok1, {})
                        preds2 = predictions_for_mcnemar.get(tok2, {})
                        gt_task = predictions_for_mcnemar["gt"].get(task)

                        flat_preds1 = get_concat_predictions(preds1, task)
                        flat_preds2 = get_concat_predictions(preds2, task)
                        if flat_preds1 and flat_preds2:
                            data1 += flat_preds1
                            data2 += flat_preds2
                            if len(flat_preds1) == len(flat_preds2):
                                repeats = len(flat_preds1) // len(gt_task)
                                correct += list(gt_task) * repeats
                            else:
                                raise ValueError(
                                    f"ERROR: Prediction length mismatch for task {task} in {tok1} vs {tok2}")
                        else:
                            print(f"ERROR: Missing predictions for task {task} in {tok1} or {tok2}")
                            continue

                if data1:
                    if do_wilcoxon:
                        stat, pval = wilcoxon(data1, data2, alternative="greater")
                    else:
                        stat, pval, _ = compute_mcnemar_statsmodels(correct, data1, data2)
                    pval_matrix.iloc[i, j] = pval
                    pval_matrix.iloc[j, i] = pval
                else:
                    pval_matrix.iloc[i, j] = pval_matrix.iloc[j, i] = None

        # Step 4: Bonferroni correction (optional)
        addition = ""
        if bonferroni:
            pval_matrix = bonferroni_correction(pval_matrix, sorted_models, fixed_nbr_correction=26 * 2)
            addition = "Bonferroni-corrected"

        test_type = "Wilcoxon" if do_wilcoxon else "McNemar"
        print(f"Pairwise {test_type} p-value matrix {addition}:")
        print(pval_matrix)

        # Step 5: Visualization
        mask = np.triu(np.ones_like(pval_matrix, dtype=bool))
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [
            (0, 'darkblue'), (0.05, 'lightblue'), (0.05, 'lightcoral'), (1, 'darkred')
        ])
        sns.set_context("paper", font_scale=2.9)
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(
            pval_matrix, mask=mask, annot=False, cmap=custom_cmap,
            fmt=".3f", vmin=0, vmax=1, cbar=show_legend
        )
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        if show_legend:
            cbar = ax.collections[0].colorbar
            cbar.set_ticks([0.05, 1])
            cbar.set_ticklabels(['0.05', '1'])
        plt.tight_layout()
        # save
        #   mkdir if not exists
        if not os.path.exists(f"_tmp/figures"):
            os.makedirs(f"_tmp/figures")
        plt.savefig(f"_tmp/figures/{tasks_name}_{tokenizer}_pval_matrix.png", dpi=300)
        plt.show()


from matplotlib.colors import LinearSegmentedColormap, FuncNorm


def forward(x):
    return np.where(x <= 0.05, (x / 0.05) * 0.5, 0.5 + ((x - 0.05) / (1 - 0.05)) * 0.5)


def inverse(x):
    return np.where(x <= 0.5, (x * 0.05) / 0.5, 0.05 + (x - 0.5) * (1 - 0.05) / 0.5)


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
    def compute_stats_across_seeds(results, task_group):
        seeds = results.keys()
        means = [
            np.mean([results[seed][task] for task in task_group])
            for seed in seeds
            if all(task in results[seed] for task in task_group)
        ]
        if means:
            return {"mean": np.mean(means), "std": np.std(means)}
        else:
            raise ValueError(f"Not all tasks in {task_group} are present in results across seeds.")

    def compute_single_stats(results, task_group):
        if all(task in results for task in task_group):
            return {"mean": np.mean([results[task] for task in task_group])}
        else:
            raise ValueError(f"Not all tasks in {task_group} are present in results.")

    mean_std_dict = {}
    for tokenizer_name, results in model_result_dict.items():
        mean_std_dict[tokenizer_name] = {}
        is_nested_by_seed = '42' in results

        task_groups = {
            "GLUE": GLUE_TASKS,
            "GLUE_TEXTFLINT": GLUE_TEXTFLINT_TASKS,
            "GLUE_MVALUE": GLUE_MVALUE_TASKS,
        }

        for group_name, task_list in task_groups.items():
            if is_nested_by_seed:
                stats = compute_stats_across_seeds(results, task_list)
            else:
                stats = compute_single_stats(results, task_list)

            if stats:
                mean_std_dict[tokenizer_name][group_name] = stats

    df = pd.DataFrame(mean_std_dict).T
    df.index.name = model_name
    print(df.to_markdown())


# def calculate_robustness_scores(model_result_dict, model_name="LR-Model"):
#     # calculate Mean and STD for LR, for ROBUSTNESS
#     mean_std_dict = {}
#     for tokenizer_name, results in model_result_dict.items():
#         mean_std_dict[tokenizer_name] = {}
#
#         # Case: Nested by seed (e.g., '42', '43', '44')
#         if '42' in results:  # includes seeds
#             # for GLUE, GLUE_TEXTFLINT, GLUE_MVALUE calculate the mean of means and std of means
#             seeds = results.keys()
#             assert len(seeds) == 3, f"Expected 3 seeds, but got {len(seeds)}"
#
#             # Compute for GLUE
#             glue_means = []
#             for seed in seeds:
#                 if all(task in results[seed] for task in GLUE_TASKS):
#                     seed_mean = np.mean([results[seed][task] for task in GLUE_TASKS])
#                     glue_means.append(seed_mean)
#             if glue_means:
#                 mean_std_dict[tokenizer_name]["GLUE"] = {
#                     "mean": np.mean(glue_means),
#                     "std": np.std(glue_means),
#                 }
#
#             # Compute for GLUE_TEXTFLINT
#             glue_tf_means = []
#             for seed in seeds:
#                 if all(task in results[seed] for task in GLUE_TEXTFLINT_TASKS):
#                     seed_mean = np.mean([results[seed][task] for task in GLUE_TEXTFLINT_TASKS])
#                     glue_tf_means.append(seed_mean)
#             if glue_tf_means:
#                 mean_std_dict[tokenizer_name]["GLUE_TEXTFLINT"] = {
#                     "mean": np.mean(glue_tf_means),
#                     "std": np.std(glue_tf_means),
#                 }
#
#             # Compute for GLUE_MVALUE
#             glue_mv_means = []
#             for seed in seeds:
#                 if all(task in results[seed] for task in GLUE_MVALUE_TASKS):
#                     seed_mean = np.mean([results[seed][task] for task in GLUE_MVALUE_TASKS])
#                     glue_mv_means.append(seed_mean)
#             if glue_mv_means:
#                 mean_std_dict[tokenizer_name]["GLUE_MVALUE"] = {
#                     "mean": np.mean(glue_mv_means),
#                     "std": np.std(glue_mv_means),
#                 }
#
#         else:  # only one seed
#             if all([key in model_result_dict[tokenizer_name] for key in GLUE_TASKS]):
#                 mean_glue = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_TASKS])
#                 mean_std_dict[tokenizer_name]["GLUE"] = {"mean": mean_glue}
#             if all([key in model_result_dict[tokenizer_name] for key in GLUE_TEXTFLINT_TASKS]):
#                 mean = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_TEXTFLINT_TASKS])
#                 mean_std_dict[tokenizer_name]["GLUE_TEXTFLINT"] = {"mean": mean}
#             if all([key in model_result_dict[tokenizer_name] for key in GLUE_MVALUE_TASKS]):
#                 mean = np.mean([model_result_dict[tokenizer_name][key] for key in GLUE_MVALUE_TASKS])
#                 mean_std_dict[tokenizer_name]["GLUE_MVALUE"] = {"mean": mean}
#
#     df = pd.DataFrame(mean_std_dict).T
#     df.index.name = model_name
#     print(df.to_markdown())


def bonferroni_correction(pval_matrix, sorted_models, fixed_nbr_correction=None):
    all_pvals = []
    for x in range(len(sorted_models)):
        for y in range(x + 1, len(sorted_models)):
            all_pvals.append(pval_matrix.iloc[x, y])

    # Apply Bonferroni
    if not fixed_nbr_correction:
        adjusted = [min(p * len(all_pvals), 1.0) for p in all_pvals]
    else:
        adjusted = [min(p * fixed_nbr_correction, 1.0) for p in all_pvals]
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


def calculate_correlation(performance_1, performance_2, no_size_difference=False, no_corpus_difference=False,
                          only_robust=False, only_sensitive=False):
    def get_mean_performance(perf_dict, tokenizer_name, task):
        """
        Get the mean performance for a task, handling both flat and seed-nested dicts.
        Returns None if the task is missing or any value is NaN.
        """
        entry = perf_dict[tokenizer_name]
        if isinstance(entry, dict) and all(isinstance(k, str) and k.isdigit() for k in entry):  # seed-nested
            task_values = [
                seed_scores[task]
                for seed_scores in entry.values()
                if task in seed_scores and not math.isnan(seed_scores[task])
            ]
            return np.mean(task_values) if task_values else None
        else:  # flat dict
            val = entry.get(task)
            return val if val is not None and not math.isnan(val) else None

    # assert not both sensitive and robust
    assert not (only_robust and only_sensitive), "Cannot filter for both robust and sensitive tasks"

    x, y = [], []

    for tokenizer_name in performance_1:
        if tokenizer_name not in performance_2:
            continue

        # Optional filtering by tokenizer size
        size = int(tokenizer_name.split("-")[-1])
        if no_size_difference and size != 32000:
            continue

        # Optional filtering by tokenizer corpus type
        corpus = tokenizer_name.split("-")[0]
        if no_corpus_difference and corpus != "mixed":
            continue

        tasks_1 = performance_1[tokenizer_name]
        tasks_2 = performance_2[tokenizer_name]

        # Determine which task names to compare
        task_names = set()

        # If performance_1 is seed-nested, gather all task names from all seeds
        if isinstance(tasks_1, dict):
            if all(isinstance(k, str) and k.isdigit() for k in tasks_1):
                # It's seed-nested, so look at all seed-level task dicts
                for seed_dict in tasks_1.values():
                    task_names.update(seed_dict.keys())
            else:
                # It's flat, just use the keys
                task_names.update(tasks_1.keys())

        # Do the same for performance_2 to ensure we get the union of task names
        if isinstance(tasks_2, dict):
            if all(isinstance(k, str) and k.isdigit() for k in tasks_2):
                for seed_dict in tasks_2.values():
                    task_names.update(seed_dict.keys())
            else:
                task_names.update(tasks_2.keys())

        if only_robust:
            task_names = set(task_names).intersection(ROBUST_TASKS)
        elif only_sensitive:
            task_names = set(task_names).intersection(SENSITIVE_TASKS)

        # Compare values across the two performance dicts
        for task in task_names:
            val1 = get_mean_performance(performance_1, tokenizer_name, task)
            val2 = get_mean_performance(performance_2, tokenizer_name, task)

            if val1 is not None and val2 is not None:
                x.append(val1)
                y.append(val2)
            else:
                print(f"Performance for {task} is NaN or missing in one of the dicts")

    if len(x) < 2:
        raise ValueError("Not enough data to compute correlation")

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
        if (task == "NUCLE") or (task == "CORE") or (task == "multi-DIALECT"):
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
                    LOG_REGRESSION[tokenizer_name][task] = classification_report["f1_micro"]
                else:
                    LOG_REGRESSION[tokenizer_name][task] = classification_report['1']["f1-score"]
    return LOG_REGRESSION


def get_BERT_performances(tasks, unique_tokenizer_paths, local_finder_addition, bert_version="base-BERT"):
    BERT_PERFORMANCE = {}
    base_out_base_path = os.path.join(local_finder_addition, "TOKENIZER/output/")
    seeds = [42, 43, 44]

    for seed in seeds:
        BERT_PATH = f"749M/steps-45000/seed-{seed}/42/"

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
            elif task in SENSITIVE_TASKS:
                task_finder_addition = f"VAR/{bert_version}/"
                results_out_base_path = os.path.join(base_out_base_path, task_finder_addition)
                if task == AV_TASK:
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
                if seed not in BERT_PERFORMANCE[tokenizer_name]:
                    BERT_PERFORMANCE[tokenizer_name][seed] = {}
                # get the BERT output for the task
                result_path = os.path.join(results_out_base_path, tokenizer_name, BERT_PATH, task_key, result_file)
                # check that path exists
                if not os.path.exists(result_path):
                    print(f"Path {result_path} does not exist")
                    continue

                if not task == AV_TASK:
                    # read in json file
                    with open(result_path, "r") as f:
                        data_dict = json.load(f)
                else:
                    data_dict = parse_av_metrics(result_path)
                if (type(performance_keys[task_key]) == str and performance_keys[task_key] in data_dict) or \
                        (performance_keys[task_key][0] in data_dict and performance_keys[task_key][1] in data_dict):
                    # get the performance from the performance keys
                    if task_key == "mnli":
                        BERT_PERFORMANCE[tokenizer_name][seed][task] = (data_dict[performance_keys[task_key][0]] +
                                                                        data_dict[
                                                                            performance_keys[task_key][1]]) / 2
                    else:
                        BERT_PERFORMANCE[tokenizer_name][seed][task] = data_dict[performance_keys[task_key]]
                else:
                    print(f"Performance key {performance_keys[task_key]} not found in {result_path}")
                    continue
    return BERT_PERFORMANCE


def get_BERT_predictions(tasks, unique_tokenizer_paths, local_finder_addition, bert_version="base-BERT"):
    BERT_PREDICTIONS = {}
    base_out_base_path = os.path.join(local_finder_addition, "TOKENIZER/output/")
    seeds = [42, 43, 44]

    for seed in seeds:
        BERT_PATH = f"749M/steps-45000/seed-{seed}/42/"
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
            elif task in SENSITIVE_TASKS:
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
                # remove hidden files
                files = [file for file in files if not os.path.basename(file).startswith(".")]
                for file in files:
                    if ((task_key in GLUE_TASKS and f"eval_dataset_{task_key}.tsv" in file) or
                            (((task_key not in GLUE_TASKS) or (task_key == "ANLI")) and "eval_dataset.tsv" in file)):
                        result_path = os.path.join(result_path, file)
                        # read in .tsv file
                        try:
                            df = pd.read_csv(result_path, sep='\t')
                        except:
                            print(f"Error reading {result_path}")
                            continue
                        # save the predictions from the "predictions" column
                        if tokenizer_name not in BERT_PREDICTIONS:
                            BERT_PREDICTIONS[tokenizer_name] = {}
                        if seed not in BERT_PREDICTIONS[tokenizer_name]:
                            BERT_PREDICTIONS[tokenizer_name][seed] = {}
                        if "gt" not in BERT_PREDICTIONS:
                            BERT_PREDICTIONS["gt"] = {}
                        if task not in BERT_PREDICTIONS["gt"]:
                            BERT_PREDICTIONS["gt"][task] = df["label"].values
                        BERT_PREDICTIONS[tokenizer_name][seed][task] = df["predictions"].values
                        break

                if ((tokenizer_name not in BERT_PREDICTIONS) or
                        (seed not in BERT_PREDICTIONS[tokenizer_name]) or
                        (task not in BERT_PREDICTIONS[tokenizer_name][seed])):
                    print(f"No predictions found for {tokenizer_name} on {task} for seed {seed} ...")
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


def parse_av_metrics(filepath):
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
