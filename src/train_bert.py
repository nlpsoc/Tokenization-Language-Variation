import argparse
import wandb
import os
import datetime

from styletokenizer.utility.custom_logger import log_and_flush

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
output_base_folder = "/shared/3/projects/hiatus/TOKENIZER_wegmann/models/tiny-BERT/"

from transformers import (DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM, AutoTokenizer,
                          Trainer, TrainingArguments, PreTrainedTokenizerFast)
from datasets import concatenate_datasets, load_dataset, load_from_disk
import torch
import tokenizers
from tokenizers import Tokenizer
from styletokenizer.utility import seed

TRAIN_DATASET_PATH = "/shared/3/projects/hiatus/TOKENIZER_wegmann/data/train-corpora/wikibook"


def load_train_dataset(test=False):
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi
    train_path = TRAIN_DATASET_PATH
    train_data = load_from_disk(train_path)["train"]
    if test:
        train_data = train_data.select(range(100))
    return train_data


def load_tokenizer(tokenizer_name):
    # check if tokenizer_name is a local path
    if os.path.exists(tokenizer_name):
        if "tokenizer.json" not in tokenizer_name:
            # add tokenizer.json to path
            tokenizer_name = os.path.join(tokenizer_name, "tokenizer.json")
        log_and_flush(f"Loading previously fitted tokenizer from local path: {tokenizer_name}")
        try:
            # tokenizer = Tokenizer.from_file(tokenizer_name)
            # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            # load the tokenizers.Tokenizer as a transformers.Tokenizer
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_name,
                                                unk_token="[UNK]",
                                                cls_token="[CLS]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                mask_token="[MASK]")
        except Exception as e:
            raise ValueError(f"Invalid tokenizer path: {tokenizer_name}")
    else:
        log_and_flush(f"Loading tokenizer from Huggingface model hub: {tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Invalid tokenizer name: {tokenizer_name}")
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    return tokenizer


def load_model(tokenizer, keep_weights=False):
    if not keep_weights:
        # Load the configuration of 'prajjwal1/bert-tiny'
        config = BertConfig.from_pretrained('prajjwal1/bert-tiny')
        log_and_flush(f"Configuration loaded: {config}")
        # Initialize the model with the configuration, this will use random weights
        model = BertForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
        log_and_flush(f"Model initialized with random weights and resized token embedding matrix: {model}")
    else:
        model = BertForMaskedLM.from_pretrained('prajjwal1/bert-tiny')  # sequence len still 512
        model.resize_token_embeddings(len(tokenizer))
        # Reinitialize the token embeddings
        model.get_input_embeddings().weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        # Verify the reset (optional)
        print(model.get_input_embeddings().weight)
    return model


# Tokenize and prepare the dataset
def tokenize_and_encode(tokenizer, examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


def main(tokenizer_name, random_seed, test=False):
    # print time
    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    tokenizer = load_tokenizer(tokenizer_name)
    log_and_flush(f"Tokenizer loaded: {tokenizer_name}")

    # set parameters
    batch_size = 32
    max_steps = 250000
    if test:
        max_steps = 100
    output_dir = os.path.join(output_base_folder, str(
        max_steps), os.path.basename(os.path.dirname(tokenizer_name)))
    log_and_flush(f"Output directory: {output_dir}")

    # REMOVED: save tokenized dataset in-between
    # tokenized_data_path = f"{output_base_folder}tokenized_data/{tokenizer_name.split('/')[-1]}-{percentage}.json"
    # tokenized_datasets = load_from_disk(tokenized_data_path)

    dataset = load_train_dataset(test=test)
    # print dataset size
    log_and_flush(f"Dataset size: {len(dataset)}")

    # Apply tokenization and encoding
    #   DANGER: map is creating cache files that potentially
    #       will be loaded even when specifying a different tokenizer later on, to be sure, delete after use
    # Tokenize the dataset with cache file names specified
    def tokenize_with_cache(dataset, tokenizer):
        cache_file_name = f"cache-{split_name}.arrow"

        # Tokenizing the dataset
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_and_encode(tokenizer, examples),
            batched=True,
            remove_columns=["text"],
            cache_file_name=cache_file_name  # Specify custom cache file name
        )

        print(f"Cache file created for {split_name}: {cache_file_name}")
        return tokenized_dataset, cache_file_name

    # tokenized_datasets = dataset.map(lambda examples: tokenize_and_encode(tokenizer, examples),
    #                                  batched=True, remove_columns=["text"])  # keep_in_memory=True
    tokenized_datasets, cache_file_name = tokenize_with_cache(dataset, "train", tokenizer)

    # # Save the tokenized dataset to disk --> REMOVED for now, check if needed
    # dataset.save_to_disk(tokenized_data_path)

    # set seed
    seed.set_global(random_seed)

    # create a randomized tiny BERT model that fits the tokenizer
    model = load_model(tokenizer, keep_weights=False)

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
        save_steps=10_000,
        save_total_limit=2,
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
        train_dataset=tokenized_datasets,
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

    log_and_flush(f"Deleting cache files at data dir {os.path.join(TRAIN_DATASET_PATH, 'train')}")


    import sys
    # add STEL folder to path
    sys.path.append('../../../STEL/src/')
    import torch

    from STEL import STEL
    from STEL.similarity import Similarity, cosine_sim
    from sentence_transformers import SentenceTransformer

    class SBERTSimilarity(Similarity):
        def __init__(self):
            super().__init__()
            self.model = SentenceTransformer(output_dir)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        def similarities(self, sentences_1, sentences_2):
            with torch.no_grad():
                sentence_emb_1 = self.model.encode(sentences_1, show_progress_bar=False)
                sentence_emb_2 = self.model.encode(sentences_2, show_progress_bar=False)
            return [cosine_sim(sentence_emb_1[i], sentence_emb_2[i]) for i in range(len(sentences_1))]

    STEL.eval_on_STEL(style_objects=[SBERTSimilarity()])


if __name__ == '__main__':
    # get cuda from command line
    parser = argparse.ArgumentParser(description="pre-train bert with specified tokenizer")
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

    log_and_flush(f"Tokenizer: {args.tokenizer}")
    log_and_flush(f"Seed: {args.seed}")
    main(tokenizer_name=args.tokenizer, random_seed=args.seed, test=args.test)

    # example call:
    # CUDA_VISIBLE_DEVICES=2 python train_bert.py --tokenizer bert-base-cased &> 24-06-09_BERT.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer meta-llama/Meta-Llama-3-8B &> 24-06-09_llama3.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer roberta-base &> 24-06-09_roberta.txt
    # CUDA_VISIBLE_DEVICES=6 python train_bert.py --tokenizer /shared/3/projects/hiatus/TOKENIZER_wegmann/tokenizer/wikipedia-gpt2-32000 --seed 42 &> 24-08-06_wiki-gpt2-32k_250000.txt
