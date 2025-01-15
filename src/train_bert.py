import argparse
import os


from styletokenizer.train_data import (UMICH_TRAIN_DATASET_PATH, UU_TRAIN_DATASET_PATH, UU_MIXED_TRAIN_DATASET_PATH,
                                       load_train_dataset)

os.environ['WANDB_CACHE_DIR'] = '/hpc/uu_cs_nlpsoc/02-awegmann/wandb_cache'
import wandb
import datetime

from styletokenizer.utility.custom_logger import log_and_flush
from styletokenizer.utility.env_variables import UU_CACHE_DIR
from transformers import (DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM, AutoTokenizer,
                          Trainer, TrainingArguments, PreTrainedTokenizerFast)
from datasets import load_from_disk
from styletokenizer.utility import seed
from styletokenizer.train_data import COUNT_PER_ROW

def load_dev_dataset(data_path=UMICH_TRAIN_DATASET_PATH, test=False):
    dev_data = load_from_disk(data_path)["dev"]
    if test:
        dev_data = dev_data.select(range(256))
    return dev_data


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


def create_tinybert_architecture(tokenizer, model_size=4):
    """
        create a tiny BERT architecture according to https://huggingface.co/prajjwal1/bert-tiny
            with random weights and resizing the embedding layer to match the tokenizer size
    :param model_size: in million
    :param tokenizer:
    :return:
    """
    if model_size == 4:
        # Load the configuration of 'prajjwal1/bert-tiny'
        config = BertConfig.from_pretrained('prajjwal1/bert-tiny')
    elif model_size == 11:
        config = BertConfig.from_pretrained('prajjwal1/bert-mini')
    elif model_size == 29:
        config = BertConfig.from_pretrained('prajjwal1/bert-small')
    elif model_size == 42:
        config = BertConfig.from_pretrained('prajjwal1/bert-medium')
    elif model_size == 110:
        config = BertConfig.from_pretrained('google-bert/bert-base-cased')
    elif model_size == 336:
        config = BertConfig.from_pretrained('google-bert/bert-large-cased')
    else:
        raise ValueError("Model size must be 4, 11, 29, 42 or 110 million, matching tiny, mini, small, medium, base "
                         "or large BERT")
    # DROPOUT in config is already set to 0.1 by default, otherwise set
    # config.hidden_dropout_prob = 0.1
    # config.attention_probs_dropout_prob = 0.1
    log_and_flush(f"Configuration loaded: {config}")
    # Initialize the model with the configuration, this will use random weights
    model = BertForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))
    log_and_flush(f"Model initialized with random weights and resized token embedding matrix: {model}")
    log_and_flush(f"Number of parameters: {model.num_parameters()}")
    return model


# Tokenize and prepare the dataset
def tokenize_and_encode(tokenizer, examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


def main(tokenizer_path, word_count, steps, random_seed, output_base_folder, data_path, batch_size=256, model_size=4,
         lr=1e-4, test=False):
    # print time
    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    tokenizer, tokenizer_name = load_tokenizer(tokenizer_path)
    log_and_flush(f"Tokenizer loaded: {tokenizer_path}")

    # set seed
    seed.set_global(random_seed)
    log_and_flush(f"Seed set to: {random_seed}")

    dataset = load_train_dataset(word_count, data_path, test=test)
    actual_word_count = len(dataset) * COUNT_PER_ROW
    log_and_flush(f"Dataset rows: {len(dataset)}")

    # set parameters
    batch_size = batch_size  # 256 * 8
    log_and_flush(f"Batch size: {batch_size}")

    # calculate the number of steps for one epoch
    epoch_steps = len(dataset) // batch_size
    log_and_flush(f"Number of steps for one epoch: {epoch_steps}")


    warm_up_steps = int(steps * 0.01)  # original BERT: 10k warm up steps over 1M steps, so 1% of steps
    if test:
        steps = 5000
        warm_up_steps = 50
    log_and_flush(f"Number of warm-up steps: {warm_up_steps}")
    log_and_flush(f"Number of steps: {steps}")
    log_and_flush(f"Number of Epochs: {steps / len(dataset) * batch_size}")

    if model_size == 4:
        bert_name = "tiny-BERT"
    elif model_size == 11:
        bert_name = "mini-BERT"
    elif model_size == 29:
        bert_name = "small-BERT"
    elif model_size == 42:
        bert_name = "medium-BERT"
    elif model_size == 110:
        bert_name = "base-BERT"
    elif model_size == 336:
        bert_name = "large-BERT"
    else:
        raise ValueError("Model size must be 4, 11, 29, 42 or 110 million, matching tiny, mini, small, medium, base"
                         " or large BERT")

    output_dir = os.path.join(output_base_folder, bert_name, tokenizer_name,
                              f"{int(actual_word_count / 1_000_000)}M",
                              f"steps-{steps}",
                              f"seed-{random_seed}")
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
    model = create_tinybert_architecture(tokenizer, model_size=model_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    now = datetime.datetime.now()
    log_and_flush(f"Current date and time : {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        save_strategy="no",
        logging_dir=output_base_folder + 'logs',
        logging_steps=1_000,
        report_to="wandb",  # Enables WandB integration
        warmup_steps=warm_up_steps,  # as in original BERT pretraining
        weight_decay=0.01,  # as in original BERT pretraining
        learning_rate=lr,
        lr_scheduler_type="linear",  # as in original BERT pretraining
        adam_beta1=0.9,  # as in original BERT pretraining
        adam_beta2=0.999,  # as in original BERT pretraining
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

    # masked LM perplexity calculation
    dev_dataset = load_dev_dataset(data_path, test=test)
    tokenized_dev_dataset = dev_dataset.map(
        lambda examples: tokenize_and_encode(tokenizer, examples),
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=False,  # Do NOT load from cache file to avoid potential conflicts
        keep_in_memory=True  # Do not create cache files at all
    )
    # Initialize the trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
    )
    # Perform evaluation
    eval_results = trainer.evaluate()
    # Extract the evaluation loss
    eval_loss = eval_results["eval_loss"]
    # Compute perplexity
    import math
    perplexity = math.exp(eval_loss)

    log_and_flush(f"Evaluation loss: {eval_loss}")
    log_and_flush(f"Perplexity: {perplexity}")


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

    # add epoch argument
    parser.add_argument("--steps", type=int, default=80000, help="number of steps to train")
    parser.add_argument("--model_size", type=int, default=4, help="in million, "
                                                                 "the number of parameters of the org tiny bert architecture")

    # add seed argument
    parser.add_argument("--seed", type=int, default=42, help="seed for random number generator")
    parser.add_argument("--test", action="store_true", help="use a tiny dataset for testing purposes")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")

    # dataset (mixed or webbook), make it a group
    train_dataset = parser.add_mutually_exclusive_group(required=True)
    train_dataset.add_argument("--mixed", action="store_true", help="Use mixed dataset.")
    train_dataset.add_argument("--webbook", action="store_true", help="Use webbook dataset.")

    # learning rate, 1e-4 from original BERT pretraining
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for training")


    # Login to WandB account (this might prompt for an API key if not logged in already)
    wandb.login(key="c042d6be624a66d40b7f2a82a76e343896608cf0")
    # Initialize a new run with a project name
    wandb.init(project="bert-tiny-pretraining", entity="annawegmann")

    args = parser.parse_args()

    if args.webbook:
        log_and_flush("Using webbook dataset")
        data_path = UU_TRAIN_DATASET_PATH
        output_base_folder = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/"
    elif args.mixed:
        log_and_flush("Using mixed dataset")
        output_base_folder = "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/"
        data_path = UU_MIXED_TRAIN_DATASET_PATH
    else:
        raise ValueError("Please specify a dataset to use")
    if args.uu:
        log_and_flush("Using UU cluster")

        os.environ["TRANSFORMERS_CACHE"] = UU_CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = UU_CACHE_DIR
    elif args.umich:
        raise NotImplementedError("UMich not implemented")
    else:
        raise ValueError("Please specify a cluster to use")


    log_and_flush(f"Dataset path: {data_path}")
    log_and_flush(f"Tokenizer: {args.tokenizer}")
    log_and_flush(f"Seed: {args.seed}")
    log_and_flush(f"Word count: {args.word_count}")
    log_and_flush(f"Steps: {args.steps}")
    log_and_flush(f"Batch size: {args.batch_size}")
    log_and_flush(f"Tiny BERT params in millions: {args.model_size}")
    main(tokenizer_path=args.tokenizer, word_count=args.word_count, steps=args.steps, random_seed=args.seed,
         output_base_folder=output_base_folder,
         data_path=data_path, batch_size=args.batch_size, model_size=args.model_size, lr=args.lr, test=args.test)

    # example call:
    # CUDA_VISIBLE_DEVICES=2 python train_bert.py --tokenizer bert-base-cased &> 24-06-09_BERT.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer meta-llama/Meta-Llama-3-8B &> 24-06-09_llama3.txt
    # CUDA_VISIBLE_DEVICES=0 python train_bert.py --tokenizer roberta-base &> 24-06-09_roberta.txt
    # CUDA_VISIBLE_DEVICES=6 python train_bert.py --tokenizer /shared/3/projects/hiatus/TOKENIZER_wegmann/tokenizer/wikipedia-gpt2-32000 --seed 42 &> 24-08-06_wiki-gpt2-32k_250000.txt
