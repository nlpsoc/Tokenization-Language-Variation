from datasets import load_from_disk

from create_webbook_sample import COUNT_PER_ROW
from utility.custom_logger import log_and_flush

UMICH_TRAIN_DATASET_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/webbook"
UU_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/webbook"


def load_train_dataset(word_count=3_300_000_000, data_path=UMICH_TRAIN_DATASET_PATH, test=False):
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi
    train_data = load_from_disk(data_path)["train"]
    # for COUNT_PER_ROW get the number of rows to sample for word_count
    nbr_rows = int(word_count // COUNT_PER_ROW)
    nbr_rows = min(nbr_rows, len(train_data))
    log_and_flush(f"Using {nbr_rows * COUNT_PER_ROW} words for pre-training.")
    # select as many rows as needed to reach the desired train_size, given one row has count COUNT_PER_ROW,
    #   this is not affected by random seed
    if test:
        train_data = train_data.select(range(256))
    else:
        train_data = train_data.select(range(nbr_rows))
    return train_data
