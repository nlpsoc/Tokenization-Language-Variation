import pandas as pd

from utility.torchtokenizer import TorchTokenizer

# load in dev_texts_roberta-base-None-30000_gyafc_0.9169461606354811.tsv
df_roberta = pd.read_csv("dev_texts_roberta-base-None-30000_gyafc_0.9169461606354811.tsv", sep="\t")
# load in dev_texts_xlm-roberta-base-None-30000_gyafc_0.8997352162400706.tsv
df_xlmroberta = pd.read_csv("dev_texts_xlm-roberta-base-None-30000_gyafc_0.8997352162400706.tsv", sep="\t")

# find entries where roberta model predicted correctly but xlm-roberta model predicted incorrectly
roberta_correct_xlmroberta_incorrect = df_roberta[(df_roberta["GT"] == df_roberta["predicted"]) &
                                                  (df_xlmroberta["GT"] != df_xlmroberta["predicted"])]

print(len(roberta_correct_xlmroberta_incorrect))

# tokenize the texts with roberta and xlm roberta attach that to the df
roberta_tokenizer = TorchTokenizer("roberta-base")
xlmroberta_tokenizer = TorchTokenizer("xlm-roberta-base")

roberta_tokens = roberta_tokenizer.tokenize(roberta_correct_xlmroberta_incorrect["text"].values.tolist())
xlmroberta_tokens = xlmroberta_tokenizer.tokenize(roberta_correct_xlmroberta_incorrect["text"].values.tolist())

# attach the tokens to the df
roberta_correct_xlmroberta_incorrect["roberta_tokens"] = roberta_tokens
roberta_correct_xlmroberta_incorrect["xlmroberta_tokens"] = xlmroberta_tokens

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# iterate over the rows
for index, row in roberta_correct_xlmroberta_incorrect.iterrows():
    print(f"Text: {row['text']}")
    print(f"GT: {row['GT']}")
    print(f"Predicted: {row['predicted']}")
    print(f"Roberta tokens    : {[tok.replace('Ġ', '▁') for tok in row['roberta_tokens']]}")
    print(f"XLM Roberta tokens: {row['xlmroberta_tokens']}")
    print("\n")

    if index == 10:
        break

