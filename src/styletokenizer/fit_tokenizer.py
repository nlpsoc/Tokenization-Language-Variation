from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
import os
import regex as re
from transformers import AutoTokenizer

# Step 1: Load the dataset
dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')

vocab_size = 100000

# train a new tokenizer based on another one
old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
def get_training_corpus():
    return (
        dataset['train'][i : i + 1000]["text"]
        for i in range(0, len(dataset['train']), 1000)
    )


training_corpus = get_training_corpus()

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_size)

dir_name = f"./llama3-tokenizer-wikitext-raw/{vocab_size}"
# make dir if it doesnt exist
os.makedirs(dir_name, exist_ok=True)
# Step 6: Save the tokenizer
tokenizer.save_pretrained(f"{dir_name}")

# load tokenizer into TorchTokenizer
from src.styletokenizer.utility.torchtokenizer import TorchTokenizer
tok = TorchTokenizer(f"{dir_name}")

# encode a test sentence
print(tok.tokenize("Hello, world!"))

