"""
    include GLUE related constants
"""
GLUE_TASKS = [
    # "mrpc",
    # "rte",
    # "wnli",
    # "cola",
    # "stsb",
    "mnli",
    "qnli",
    "sst2",
    "qqp",
]

HUGGINGFACE_HANDLE = "nyu-mll/glue"

LOCAL_PREFIX = "/Users/anna/sftp_mount/hpc_disk2/02-awegmann/TOKENIZER"
PATH_PREFIX = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER"

GLUE_TEXTFLINT = {
    "mnli-textflint": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/MNLI/mnli_train_textflint.csv",
        "dev": [f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/MNLI/mnli_val_matched_textflint.csv",
                f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/MNLI/mnli_val_mismatched_textflint.csv"]},
    "sst2-textflint": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/SST2/sst2_train_textflint.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/SST2/sst2_dev_textflint.csv"},
    "qqp-textflint": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/QQP/qqp_train_textflint.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/QQP/qqp_val_textflint.csv"},
    "qnli-textflint": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/QNLI/qnli_train_textflint.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/GLUE_textflint/QNLI/qnli_val_textflint.csv"},
}

GLUE_MVALUE = {
    "sst2-mVALUE": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/sst2_multi/train.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/sst2_multi/validation.csv",
    },
    "qqp-mVALUE": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/qqp_multi/train.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/qqp_multi/validation.csv",
    },
    "mnli-mVALUE": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/mnli_multi/train.csv",
        "dev": [
            f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/mnli_multi/validation_matched.csv",
            f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/mnli_multi/validation_mismatched.csv"]
    },
    "qnli-mVALUE": {
        "train": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/qnli_multi/train.csv",
        "dev": f"{PATH_PREFIX}/data/eval-corpora/multi-VALUE/qnli_multi/validation.csv",
    },
}

GLUE_TEXTFLINT_TASKS = list(GLUE_TEXTFLINT.keys())
GLUE_MVALUE_TASKS = list(GLUE_MVALUE.keys())

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
