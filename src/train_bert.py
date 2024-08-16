import argparse
import wandb
import os
import datetime

from styletokenizer.utility.custom_logger import log_and_flush

UMICH_CACHE_DIR = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
UU_CACHE_DIR = "/hpc/uu_cs_nlpsoc/02-awegmann/huggingface"

UMICH_TRAIN_DATASET_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/wikibook"
UU_TRAIN_DATASET_PATH = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/train-corpora/wikibook"


def load_train_dataset(word_count=3_300_000_000, data_path=UMICH_TRAIN_DATASET_PATH, test=False):
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi
    train_data = load_from_disk(data_path)["train"]
    # for COUNT_PER_ROW get the number of rows to sample for word_count
    nbr_rows = int(word_count // COUNT_PER_ROW)
    nbr_rows = min(nbr_rows, len(train_data))
    log_and_flush(f"Using {nbr_rows*COUNT_PER_ROW} words for pre-training.")
    # select as many rows as needed to reach the desired train_size, given one row has count COUNT_PER_ROW
    if test:
        train_data = train_data.select(range(100))
    else:
        train_data = train_data.select(range(nbr_rows))
    return train_data


def load_tokenizer(tokenizer_path):
    """
        Load tokenizer from local path or, if it does not exist,from the Huggingface model hub
        Attention: for local path it expects the tokenizers tokenizer.json file
    :param tokenizer_path:
    :return:
    """
    # check if tokenizer_name is a local path
    if os.path.exists(tokenizer_path):
        if "tokenizer.json" not in tokenizer_path:
            # add tokenizer.json to path
            tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
        log_and_flush(f"Loading previously fitted tokenizer from local path: {tokenizer_path}")
        try:
            # loading tokenizers tokenizer as a transformers tokenizer
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path,
                                                unk_token="[UNK]",
                                                cls_token="[CLS]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                mask_token="[MASK]")
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
        except Exception as e:
            raise ValueError(f"Invalid tokenizer path: {tokenizer_path}")
    else:
        log_and_flush(f"Loading tokenizer from Huggingface model hub: {tokenizer_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer_name = tokenizer_path
        except Exception as e:
            raise ValueError(f"Invalid tokenizer name: {tokenizer_path}")
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    return tokenizer, tokenizer_name


def create_tinybert_architecture(tokenizer):
    """
        create a tiny BERT architecture according to https://huggingface.co/prajjwal1/bert-tiny
            with random weights and resizing the embedding layer to match the tokenizer size
    :param tokenizer:
    :return:
    """
    # Load the configuration of 'prajjwal1/bert-tiny'
    config = BertConfig.from_pretrained('prajjwal1/bert-tiny')
    log_and_flush(f"Configuration loaded: {config}")
    # Initialize the model with the configuration, this will use random weights
    model = BertForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))
    log_and_flush(f"Model initialized with random weights and resized token embedding matrix: {model}")
    return model


# Tokenize and prepare the dataset
def tokenize_and_encode(tokenizer, examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


def main(tokenizer_path, word_count, random_seed, output_base_folder, data_path, test=False):
    # print time
    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    tokenizer, tokenizer_name = load_tokenizer(tokenizer_path)
    log_and_flush(f"Tokenizer loaded: {tokenizer_path}")

    # set seed
    seed.set_global(random_seed)
    log_and_flush(f"Seed set to: {random_seed}")

    dataset = load_train_dataset(word_count, data_path, test=test)
    log_and_flush(f"Dataset size: {len(dataset)}")

    # set parameters
    batch_size = 32

    # calculate the number of steps for one epoch
    epoch_steps = len(dataset) // batch_size
    log_and_flush(f"Number of steps for one epoch: {epoch_steps}")

    # set number of steps to a maximum of 20 epochs
    max_steps = min(20 * epoch_steps, 250000)
    if test:
        max_steps = 100
    log_and_flush(f"Maximum number of steps: {max_steps}")
    log_and_flush(f"Number of Epochs: {max_steps / len(dataset) * batch_size}")

    output_dir = os.path.join(output_base_folder, tokenizer_name, f"{int(word_count/1_000_000)}M", f"steps-{max_steps}", f"seed-{random_seed}")
    log_and_flush(f"Output directory: {output_dir}")


    # Apply tokenization
    #   DANGER: map is creating cache files that potentially
    #       will be loaded even when specifying a different tokenizer later on, to be sure, delete after use
    # Tokenize the dataset with cache file names specified
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_encode(tokenizer, examples),
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,  # Do NOT load from cache file to avoid potential conflicts
        keep_in_memory=True  # Do not create cache files at all
        # (potentially have to remove later, depending on RAM impact)
    )

    # initialize a tiny BERT model that fits the tokenizer size
    model = create_tinybert_architecture(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_dir=output_base_folder + 'logs',
        logging_steps=500,
        report_to="wandb",  # Enables WandB integration
        warmup_steps=1000,
        weight_decay=0.01,
        learning_rate=4e-4,
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start pretraining
    trainer.train()

    # print time
    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # create dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save the trained model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log_and_flush(f"Model saved to: {output_dir}")

    log_and_flush(f"Deleting cache files at data dir {os.path.join(UMICH_TRAIN_DATASET_PATH, 'train')}")


if __name__ == '__main__':
    # get cuda from command line
    parser = argparse.ArgumentParser(description="pre-train bert with specified tokenizer")

    # either uu or umich basefolder
    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    # Add options to the mutually exclusive group
    group.add_argument("--uu", action="store_true", help="Use UMich cluster.")
    group.add_argument("--umich", action="store_true", help="Use UU cluster.")

    # number of words
    parser.add_argument("--word_count", type=int, default=3_300_000_000,
                        help="number of words to train on")

    # load the tokenizer, either by downloading it from huggingface hub, or calling it from the local path
    #   /shared/3/project/hiatus/TOKENIZER_wegmann/tokenizer
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="tokenizer to use")

    # add seed argument
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    parser.add_argument("--test", action="store_true", help="use a tiny dataset for testing purposes")

    # Login to WandB account (this might prompt for an API key if not logged in already)
    wandb.login(key="c042d6be624a66d40b7f2a82a76e343896608cf0")
    # Initialize a new run with a project name
    wandb.init(project="bert-tiny-pretraining", entity="annawegmann")

    args = parser.parse_args()

    if args.uu:
        log_and_flush("Using UU cluster")
        output_base_folder = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/"
        train_path = UU_TRAIN_DATASET_PATH
        os.environ["TRANSFORMERS_CACHE"] = UU_CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = UU_CACHE_DIR
    elif args.umich:
        log_and_flush("Using UMich cluster")
        output_base_folder = "/shared/3/projects/hiatus/TOKENIZER_wegmann/models/tiny-BERT/"
        train_path = UMICH_TRAIN_DATASET_PATH
        os.environ["TRANSFORMERS_CACHE"] = UMICH_CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = UMICH_CACHE_DIR
    else:
        raise ValueError("Please specify a cluster to use")

    from transformers import (DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM, AutoTokenizer,
                              Trainer, TrainingArguments, PreTrainedTokenizerFast)
    from datasets import load_from_disk
    from styletokenizer.utility import seed
    import torch
    from create_webbook_sample import COUNT_PER_ROW

    log_and_flush(f"Tokenizer: {args.tokenizer}")
    log_and_flush(f"Seed: {args.seed}")
    main(tokenizer_path=args.tokenizer, word_count=args.word_count, random_seed=args.seed, output_base_folder=output_base_folder,
         data_path=train_path, test=args.test)

    # example call:
    # CUDA_VISIBLE_DEVICES=2 python train_bert.py --tokenizer bert-base-cased &> 24-06-09_BERT.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer meta-llama/Meta-Llama-3-8B &> 24-06-09_llama3.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer roberta-base &> 24-06-09_roberta.txt
    # CUDA_VISIBLE_DEVICES=6 python train_bert.py --tokenizer /shared/3/projects/hiatus/TOKENIZER_wegmann/tokenizer/wikipedia-gpt2-32000 --seed 42 &> 24-08-06_wiki-gpt2-32k_250000.txt
