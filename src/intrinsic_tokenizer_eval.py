import itertools
import os
from styletokenizer.utility.custom_logger import log_and_flush
from eval_tokenizer import calc_renyi_efficency_from_generator, calc_seq_len_from_generator, calc_avg_tok_from_generator
from run_glue import task_to_keys as glue_task_to_keys
from styletokenizer.utility.env_variables import set_cache

set_cache()
from datasets import load_dataset, load_from_disk

from styletokenizer.utility.tokenizer_vars import OUT_PATH
from styletokenizer.fitting_corpora import CORPORA_TWITTER, CORPORA_WIKIPEDIA, CORPORA_MIXED
from styletokenizer.utility.preptraining_corpora import CORPORA_WEBBOOK

BASE_TOKENIZER_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/"
BASE_TOKENIZER_PATH = OUT_PATH

TOKENIZER_PATHS = [
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "twitter-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "wikipedia-gpt2-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-ws-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-llama3-32000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-500/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-1000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-2000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-4000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-8000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-16000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-64000/tokenizer.json"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-128000/tokenizer.json"),
]

GLUE_TASKS = [
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
]

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


def load_eval_data(task_name_or_hfpath=None, csv_file=None, split=None):
    """
        loading the different eval datasets in all different forms (from huggingface datasets,
         locally with datasets, or csv)
    :param task_name_or_hfpath:
    :param csv_file:
    :return:
    """
    # load dev set of the datasets
    if task_name_or_hfpath is not None:
        if os.path.exists(task_name_or_hfpath):
            if split:
                raw_datasets = load_from_disk(os.path.join(task_name_or_hfpath, split))
            else:
                raw_datasets = load_from_disk(task_name_or_hfpath)
            # set task name to last folder in path
            task_name_or_hfpath = os.path.basename(os.path.normpath(task_name_or_hfpath))
        else:
            # GLUE: MRPC, CoLA, SST-2, QNLI, QQP, RTE, WNLI, STS-B
            raw_datasets = load_dataset(
                "nyu-mll/glue",
                task_name_or_hfpath
            )
        log_and_flush(f"Dataset loaded: {raw_datasets}")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"validation": csv_file}  # should also work with list of csv files
        if csv_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files
            )
    return raw_datasets


def main():
    # task_name_or_hfpath = "mnli"
    for task_name_or_hfpath in (FITTING_CORPORA + GLUE_TASKS + VALUE_PATHS + VARIETIES_TASKS):
        split=None
        if task_name_or_hfpath in VARIETIES_TASK_DICT.keys():
            task = task_name_or_hfpath
            task_name_or_hfpath = VARIETIES_TASK_DICT[task_name_or_hfpath]
            task_to_keys = VARIETIES_to_keys
        elif task_name_or_hfpath in FITTING_CORPORA:
            task_to_keys = {"mixed": ["text"], "twitter": ["text"], "wikipedia": ["text"], "webbook": ["text"]}
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))  # doesn't really matter which one
            split = "dev"
        else:
            task = os.path.basename(os.path.normpath(task_name_or_hfpath))
            task_to_keys = glue_task_to_keys
        raw_datasets = load_eval_data(task_name_or_hfpath, split=split)
        try:
            eval_dataset = raw_datasets["validation_matched" if task == "mnli" else "validation"]
        except KeyError:  # some of the datasets are not provided in split form
            eval_dataset = raw_datasets
        sentence_keys = task_to_keys[task]
        for tokenizer_path in TOKENIZER_PATHS:
            text_generator = (" ".join(example[text_key] for text_key in sentence_keys) for example in eval_dataset)
            text_generator, t_gen1, t_gen2, t_gen3 = itertools.tee(text_generator, 4)
            log_and_flush(f"\n{task_name_or_hfpath} - {tokenizer_path}")
            log_and_flush(f"Renyi Efficency: {calc_renyi_efficency_from_generator(t_gen1, tokenizer_path)}")
            log_and_flush(f"Avg Seq Len: {calc_seq_len_from_generator(t_gen2, tokenizer_path)}")
            log_and_flush(f"Avg # Toks/Words + Seq Len (slow impl.): "
                          f"{calc_avg_tok_from_generator(t_gen3, tokenizer_path)}")



if __name__ == "__main__":
    main()
