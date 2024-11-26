import argparse
import itertools
import os

from styletokenizer.glue import GLUE_TASKS
from styletokenizer.utility.custom_logger import log_and_flush
from eval_tokenizer import calc_renyi_efficency_from_generator, calc_seq_len_from_generator
from run_glue import task_to_keys as glue_task_to_keys
from styletokenizer.utility.env_variables import set_cache
from styletokenizer.tokenizer import TOKENIZER_PATHS
from styletokenizer.utility.datasets_helper import load_data
from tokenizer import BASE_TOKENIZER_PATH

set_cache()

from styletokenizer.utility.tokenizer_vars import OUT_PATH
from styletokenizer.fitting_corpora import CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED
from styletokenizer.utility.preptraining_corpora import CORPORA_WEBBOOK

BASE_TOKENIZER_PATH = OUT_PATH

VALUE_BASE = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/value/"
VALUE_PATHS = [
    os.path.join(VALUE_BASE, glue_task) for glue_task in GLUE_TASKS
]

VARIETIES_TASK_DICT = {
    "sadiri": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation",
    "stel": ["/home/uu_cs_nlpsoc/awegmann/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv",
             "/home/uu_cs_nlpsoc/awegmann/STEL/characteristics/quad_questions_char_contraction.tsv",
             "/home/uu_cs_nlpsoc/awegmann/STEL/characteristics/quad_questions_char_substitution.tsv"],
    "age": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/blogcorpus/validation.csv",
    "CORE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/multilabel_dev.tsv",
    "CGLU": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Varieties/CGLUv5.2/dev.csv",
    "GYAFC": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GYAFC/dev.csv",
    "DIALECT": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/Dialect/combined_validation.csv",
}
VARIETIES_to_keys = {
    "sadiri": ("query_text", "candidate_text"),
    "stel": ["Anchor 1", "Anchor 2", "Alternative 1.1", "Alternative 1.2"],
    "age": ["text"],
    "CORE": ["text"],
    "CGLU": ["Text"],
    "GYAFC": ["text"],
    "DIALECT": ["text"],
}
VARIETIES_TASKS = list(VARIETIES_TASK_DICT.keys())

FITTING_CORPORA = [CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED, CORPORA_WEBBOOK]


def main(output_path=None):
    # task_name_or_hfpath = "mnli"
    # create a result dataframe with
    # task_name, tokenizer_path, renyi_eff_2.5, renyi_eff_3.0, avg_seq_len
    result_dict = {
        "task_name": [],
        "tokenizer_path": [],
        "renyi_eff_2.5": [],
        "renyi_eff_3.0": [],
        "avg_seq_len": []
    }
    for task_name_or_hfpath in (VARIETIES_TASKS + FITTING_CORPORA + GLUE_TASKS + VALUE_PATHS):
        split = None
        csv_file = False
        if task_name_or_hfpath in VARIETIES_TASK_DICT.keys():
            task = task_name_or_hfpath
            task_name_or_hfpath = VARIETIES_TASK_DICT[task_name_or_hfpath]
            task_to_keys = VARIETIES_to_keys
            if task != "sadiri":
                csv_file = True
        elif task_name_or_hfpath in FITTING_CORPORA:
            task_to_keys = {"mixed": ["text"], "twitter": ["text"], "wikipedia": ["text"], "webbook": ["text"]}
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))  # doesn't really matter which one
            split = "test"
        else:
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))
            task_to_keys = glue_task_to_keys
        if csv_file:
            raw_datasets = load_data(csv_file=task_name_or_hfpath, split=split)
        else:
            raw_datasets = load_data(task_name_or_hfpath, split=split)
        try:
            eval_dataset = raw_datasets["validation_matched" if task == "mnli" else "validation"]
        except KeyError:  # some of the datasets are not provided in split form
            eval_dataset = raw_datasets
        sentence_keys = task_to_keys[task]
        for tokenizer_path in TOKENIZER_PATHS:
            text_generator = (" ".join(example[text_key] for text_key in sentence_keys if text_key is not None)
                              for example in eval_dataset)
            text_generator, t_gen1, t_gen2, t_gen3 = itertools.tee(text_generator, 4)
            log_and_flush(f"\n{task_name_or_hfpath} - {tokenizer_path}")
            renyi_25 = calc_renyi_efficency_from_generator(t_gen1, tokenizer_path, power=2.5)
            log_and_flush(f"Renyi Efficency (2.5): "
                          f"{renyi_25}")
            renyi_30 = calc_renyi_efficency_from_generator(t_gen2, tokenizer_path, power=3.0)
            log_and_flush(f"Renyi Efficency (3.0): "
                          f"{renyi_30}")
            seq_len = calc_seq_len_from_generator(t_gen3, tokenizer_path)
            log_and_flush(f"Avg Seq Len: {seq_len}")
            result_dict["task_name"].append(task)
            result_dict["tokenizer_path"].append(tokenizer_path)
            result_dict["renyi_eff_2.5"].append(renyi_25)
            result_dict["renyi_eff_3.0"].append(renyi_30)
            result_dict["avg_seq_len"].append(seq_len)
    import pandas as pd
    result_df = pd.DataFrame(result_dict)
    if output_path:
        result_df.to_csv(os.path.join(output_path, "intrinsic_tokenizer_eval_results.csv"), sep="\t")
    else:
        result_df.to_csv("eval_results.csv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', default=None, type=str)
    args = parser.parse_args()
    main(output_path=args.output_path)
