"""
    paths to tasks sensitive to language variation
"""
VARIETIES_DEV_DICT = {
    "NUCLE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/NUCLE/multilabel/dev.csv",
    "CORE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/multiclass_dev_stratified.tsv",
    "sadiri": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/validation/"
                                "validation.csv",
    "PAN": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/PAN/PAN-hard_validation.csv",
    "multi-DIALECT": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-DIALECT/validation.csv",
}
VARIETIES_TRAIN_DICT = {
    "NUCLE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/NUCLE/multilabel/train.csv",
    "sadiri": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/train/"
                                  "train.csv",
    "CORE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/CORE/multiclass_train_stratified.tsv",
    "PAN": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/PAN/PAN-hard_train.csv",
    "multi-DIALECT": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-DIALECT/train.csv",
}
VARIETIES_TEST_DICT = {
    "sadiri": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/down_1_shuffle/test/test.csv",
    "NUCLE": "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/NUCLE/multilabel/test.csv",
}
VARIETIES_to_keys = {
    "CORE": ["text"],
    "multi-DIALECT": ["text"],
    "sadiri": ("query_text", "candidate_text"),
    "NUCLE": ["sentence"],  # ["sentence1", "sentence2"],
    "PAN": ("text 1", "text 2"),
}
VARIETIES_to_labels = {
    "CORE": "genre",
    "multi-DIALECT": "label",
    "sadiri": "label",
    "NUCLE": "label",  # "Error Overlap",
    "PAN": "Author Change",
}
SENSITIVE_TASKS = list(VARIETIES_DEV_DICT.keys())

