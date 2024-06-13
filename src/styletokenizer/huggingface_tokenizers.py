ROBERTA = "roberta-base"
XLMROBERTA = "xlm-roberta-base"
T5 = "google-t5/t5-base"
LLAMA3 = "meta-llama/Meta-Llama-3-70B"
LLAMA2 = "meta-llama/Llama-2-70b-hf"
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
BERT_CASED = "bert-base-cased"
BERT_UNCASED = "bert-base-uncased"
MBERT = "google-bert/bert-base-multilingual-cased"

HUGGINGFACE_TOKENIZERS = [ROBERTA, XLMROBERTA, BERT_CASED,
                          LLAMA2, LLAMA3, T5, MIXTRAL, BERT_UNCASED, MBERT]

BASEFOLDER = "/Users/anna/Documents/git projects.nosync/StyleTokenizer/src/styletokenizer"
# TRAINED_TOKENIZERS = [f"{BASEFOLDER}/llama3-tokenizer-wiki-20GB-raw/{vocab_size}" for vocab_size in [500, 8000, 16000, 32000, 64000]]
# TRAINED_TOKENIZERS = [f"{BASEFOLDER}/llama3-tokenizer-wiki-20GB-raw/{vocab_size}" for vocab_size in [2000, 4000, 128000]]
# RAINED_TOKENIZERS = [f"{BASEFOLDER}/llama3-tokenizer-wiki-20GB-raw/{vocab_size}" for vocab_size in [1000, 246000, 512000]]
TRAINED_TOKENIZERS = [f"{BASEFOLDER}/llama3-tokenizer-twitter-raw/{vocab_size}" for vocab_size in [32000]]