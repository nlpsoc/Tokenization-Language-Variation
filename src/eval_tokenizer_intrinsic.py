import os

from styletokenizer.utility.custom_logger import log_and_flush
from fit_tokenizer import OUT_PATH
from utility.tokenizer_vars import PRE_TOKENIZER, VOCAB_SIZE, FITTING_CORPORA
from eval_tokenizer import calc_renyi_efficency_from_generator, calc_seq_len, calc_avg_tok_from_generator, main, calc_precentile_freq
from styletokenizer.utility.env_variables import set_cache

set_cache()
from datasets import load_dataset, load_from_disk

BASE_TOKENIZER_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/"

TOKENIZER_PATHS = [
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-32000"),
    os.path.join(BASE_TOKENIZER_PATH, "twitter-gpt2-32000"),
    os.path.join(BASE_TOKENIZER_PATH, "wikipedia-gpt2-32000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-ws-32000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-llama3-32000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-500"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-1000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-2000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-4000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-8000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-16000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-64000"),
    os.path.join(BASE_TOKENIZER_PATH, "mixed-gpt2-128000"),
]

GLUE_TASKS = [
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
]
from run_glue import task_to_keys

def main(task_name_or_hfpath=None, csv_file=None):
    task = None
    # load dev set of the datasets
    if task_name_or_hfpath is not None:
        if os.path.exists(task_name_or_hfpath):
            raw_datasets = load_from_disk(task_name_or_hfpath)
            # set task name to last folder in path
            task_name_or_hfpath = os.path.basename(os.path.normpath(task_name_or_hfpath))
            task_name_or_hfpath = task_name_or_hfpath
        else:
            # GLUE: MRPC, CoLA, SST-2, QNLI, QQP, RTE, WNLI, STS-B
            raw_datasets = load_dataset(
                "nyu-mll/glue",
                task_name_or_hfpath
            )
        log_and_flush(f"Dataset loaded: {raw_datasets}")
        task = task_name_or_hfpath
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"validation": csv_file}
        if csv_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files
            )
        task = csv_file

    if task in GLUE_TASKS:
        eval_dataset = raw_datasets["validation_matched" if task_name_or_hfpath == "mnli" else "validation"]
        sentence1_key, sentence2_key = task_to_keys[task_name_or_hfpath]
        text_generator = (example[sentence1_key] + " " + example[sentence2_key] for example in eval_dataset)
    else:
        raise ValueError(f"Invalid task: {task}")

    for tokenizer_path in TOKENIZER_PATHS:
        print(f"\n{task} - {tokenizer_path}")
        print(calc_renyi_efficency_from_generator(tokenizer_path, text_generator))
        print(calc_avg_tok_from_generator(tokenizer_path, text_generator))


if __name__ == "__main__":
    main()
