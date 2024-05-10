import pandas as pd

from styletokenizer.utility import bpe

from styletokenizer.utility import seed, POS, wordpiece

seed.set_global()

from styletokenizer.logistic_regression import TextClassifier
from styletokenizer.tokenizer import TorchTokenizer
from styletokenizer.utility import gyafc
import random
from utility.gyafc import to_classification_data


def shuffle_lists_in_unison(list1, list2):
    """ generated CoPilot, April 29 2024

    """
    # Create a list of indices
    indices = list(range(len(list1)))
    # Shuffle the indices
    random.shuffle(indices)
    # Use the shuffled indices to rearrange both lists
    list1 = [list1[i] for i in indices]
    list2 = [list2[i] for i in indices]
    return list1, list2


def main(tok=None, ngram=None, pre_tokenizer=None, vocab_size=30000, data="gyafc"):
    if data == "gyafc":
        # Load the training data
        train_data = gyafc.load_train_data()
        train_labels, train_texts = to_classification_data(train_data)
    elif data == "UAAVE":
        # load in UAAVE_org-wiki_30000.txt
        train_data_0 = "../../data/multivalue/UAAVE_org-wiki_30000_train.txt"
        train_data_1 = "../../data/multivalue/UAAVE_transformed-wiki_30000_train.txt"
        train_labels, train_texts = load_parallel_data(train_data_0, train_data_1)
    elif data == "definite_abstract":
        train_data_0 = "../../data/multivalue/definite_abstract_org-wiki_30000_train.txt"
        train_data_1 = "../../data/multivalue/definite_abstract_transformed-wiki_30000_train.txt"
        train_labels, train_texts = load_parallel_data(train_data_0, train_data_1)
    else:
        raise ValueError(f"Invalid data: {data}")
    #   shuffle both lists the same way
    train_texts, train_labels = shuffle_lists_in_unison(train_texts, train_labels)

    # Set the Tokenizer
    print(f"Setting tokenizer to {tok}")
    if not tok:
        classifier = TextClassifier(ngram=ngram)
    elif tok == "POS":
        classifier = TextClassifier(tokenizer=POS.tokenize, ngram=ngram)
    elif tok == "bpe":
        bpe_tokenizer = bpe.BPETokenizer(vocab_size=vocab_size, pre_tokenizer=pre_tokenizer)
        bpe_tokenizer.train(train_texts)
        classifier = TextClassifier(tokenizer=bpe_tokenizer.tokenize, ngram=ngram)
    elif tok == "wordpiece":
        wordpiece_tokenizer = wordpiece.WordpieceTokenizer(vocab_size=vocab_size, pre_tokenizer=pre_tokenizer)
        wordpiece_tokenizer.train(train_texts)
        classifier = TextClassifier(tokenizer=wordpiece_tokenizer.tokenize, ngram=ngram)
    elif type(tok) == str:
        tokenizer = TorchTokenizer(tok)
        classifier = TextClassifier(tokenizer=tokenizer.tokenize, ngram=ngram)
    else:
        raise ValueError(f"Invalid tokenizer: {tok}")

    classifier.fit_vectorizer(train_texts)
    classifier.get_most_frequent_tokens(train_texts)

    # Fit the TextClassifier model on the training data
    classifier.fit(train_texts, train_labels)

    # Load the development data
    if data == "gyafc":
        dev_data = gyafc.load_dev_data()
        dev_labels, dev_texts = to_classification_data(dev_data)
    elif data == "UAAVE":
        dev_data_0 = "../../data/multivalue/UAAVE_org-wiki_30000_dev.txt"
        dev_data_1 = "../../data/multivalue/UAAVE_transformed-wiki_30000_dev.txt"
        dev_labels, dev_texts = load_parallel_data(dev_data_0, dev_data_1)
    elif data == "definite_abstract":
        dev_data_0 = "../../data/multivalue/definite_abstract_org-wiki_30000_dev.txt"
        dev_data_1 = "../../data/multivalue/definite_abstract_transformed-wiki_30000_dev.txt"
        dev_labels, dev_texts = load_parallel_data(dev_data_0, dev_data_1)
    else:
        raise ValueError(f"Invalid data: {data}")



    # Evaluate the model on the development data
    score, predicted = classifier.score(dev_texts, dev_labels)

    # Print the performance of the model
    print(f'Model performance on development set: {score}')
    print(f'Number of features: {len(classifier.get_coefficients())}')
    print(f'Formal set to 1 in classification')
    classifier.print_extreme_coefficients()

    # save predictions and text in tsv file using pandas
    df = pd.DataFrame(list(zip(dev_texts, dev_labels, predicted)), columns=["text", "GT", "predicted"])
    df.to_csv(f"dev_texts_{tok}-{pre_tokenizer}-{vocab_size}_{data}_{score}.tsv", sep="\t", index=False)


def load_parallel_data(train_data_0, train_data_1):
    with open(train_data_0, 'r') as file:
        # Read each line, strip the newline character, and store it in a list
        lines_0 = [line.strip() for line in file.readlines()]
        labels_0 = [0 for _ in range(len(lines_0))]
    with open(train_data_1, 'r') as file:
        # Read each line, strip the newline character, and store it in a list
        lines_1 = [line.strip() for line in file.readlines()]
        labels_1 = [1 for _ in range(len(lines_1))]
    train_texts = lines_0 + lines_1
    train_labels = labels_0 + labels_1
    return train_labels, train_texts


roberta = "roberta-base"
t5="google-t5/t5-base"
xlmroberta = "xlm-roberta-base"
llama27b = "meta-llama/Llama-2-7b-hf"
mixtreal = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llama3 = "meta-llama/Meta-Llama-3-70B"
llama2 = "meta-llama/Llama-2-70b-hf"
anna = "AnnaWegmann/Style-Embedding"
gpt2 = "openai-community/gpt2"
bertcased = "bert-base-cased"
ws = None
pos = "POS"


main(tok="bpe", vocab_size=1000, pre_tokenizer=None)  # , ngram=3

# main(tok="bpe", vocab_size=10000, pre_tokenizer="byte", data="definite_abstract", ngram=2)  # , ngram=3
# main(tok="bpe", pre_tokenizer="meta", data="UAAVE")  # , ngram=3
# main(tok=llama2, data="definite_abstract")  # , ngram=3
# main(tok="bpe", pre_tokenizer="meta", data="definite_abstract")  # , ngram=3


# main(tok="wordpiece", pre_tokenizer="byte", vocab_size=20000)  # , ngram=3

