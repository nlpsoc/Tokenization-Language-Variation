import argparse
import wandb
import os
from transformers import DataCollatorForLanguageModeling

cache_dir = "/shared/3/projects/hiatus/EVAL_wegmann/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
output_folder = "/shared/3/projects/hiatus/EVAL_wegmann/tiny-BERT/"


def load_dataset():
    # loading dataset, following https://huggingface.co/blog/pretraining-bert#4-pre-train-bert-on-habana-gaudi
    from datasets import concatenate_datasets, load_dataset

    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki])
    return raw_datasets


def load_tokenizer(tokenizer_name):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'mask_token': '[MASK]'})
    return tokenizer


def load_model(tokenizer):
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained('prajjwal1/bert-tiny')  # sequence len still 512
    model.resize_token_embeddings(len(tokenizer))
    return model


# Tokenize and prepare the dataset
def tokenize_and_encode(tokenizer, examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)


def main(tokenizer_name):
    # print time
    import datetime
    now = datetime.datetime.now()
    print("Current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    dataset = load_dataset()
    tokenizer = load_tokenizer(tokenizer_name)
    model = load_model(tokenizer)
    # Apply tokenization and encoding
    tokenized_datasets = dataset.map(lambda examples: tokenize_and_encode(tokenizer, examples),
                                     batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    from transformers import Trainer, TrainingArguments

    # print time
    now = datetime.datetime.now()
    print("Current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_folder + "bert-tiny-pretrained",
        overwrite_output_dir=True,
        max_steps=100000,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=output_folder + 'logs',
        logging_steps=500,
        report_to="wandb",  # Enables WandB integration
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    # Start pretraining
    trainer.train()

    # print time
    now = datetime.datetime.now()
    print("Current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    # Save the trained model
    trainer.save_model(output_folder + "bert-tiny-pretrained/" + tokenizer_name)
    tokenizer.save_pretrained(output_folder + "bert-tiny-pretrained" + tokenizer_name)

    import sys
    # add STEL folder to path
    sys.path.append('../../STEL/src/')
    import torch

    from STEL import STEL
    from STEL.similarity import Similarity, cosine_sim
    from sentence_transformers import SentenceTransformer

    class SBERTSimilarity(Similarity):
        def __init__(self):
            super().__init__()
            self.model = SentenceTransformer(output_folder + "bert-tiny-pretrained/" + tokenizer_name)
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
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="tokenizer to use")

    # Login to WandB account (this might prompt for an API key if not logged in already)
    wandb.login(key="c042d6be624a66d40b7f2a82a76e343896608cf0")
    # Initialize a new run with a project name
    wandb.init(project="bert-tiny-pretraining", entity="annawegmann")

    args = parser.parse_args()
    main(tokenizer_name=args.tokenizer)
