import pandas as pd

from utility.torchtokenizer import TorchTokenizer

def print_text_and_tokenization(df, tokenizer1, tokenizer2):
    print(f"Number of inconsistent predictions {len(df)}")
    # tokenize the texts with roberta and xlm roberta attach that to the df
    roberta_tokens = tokenizer1.tokenize(df["text"].values.tolist())
    xlmroberta_tokens = tokenizer2.tokenize(df["text"].values.tolist())
    # attach the tokens to the df
    df["roberta_tokens"] = roberta_tokens
    df["xlmroberta_tokens"] = xlmroberta_tokens
    # iterate over the rows
    for i, row in df.iterrows():
        print(f"Text: {row['text']}")
        print(f"GT: {row['GT']}")
        print(f"Predicted: {row['predicted']}")
        print(f"Roberta tokens    : {[tok.replace('Ġ', '▁') for tok in row['roberta_tokens']]}")
        print(f"XLM Roberta tokens: {row['xlmroberta_tokens']}")
        print("\n")

        if i == 10:
            break

# load in dev_texts_roberta-base-None-30000_gyafc_0.9169461606354811.tsv
df_roberta = pd.read_csv("dev_texts_roberta-base-None-30000_gyafc_0.9169461606354811.tsv", sep="\t")
# load in dev_texts_xlm-roberta-base-None-30000_gyafc_0.8997352162400706.tsv
df_xlmroberta = pd.read_csv("dev_texts_xlm-roberta-base-None-30000_gyafc_0.8997352162400706.tsv", sep="\t")

# find entries where roberta model predicted correctly but xlm-roberta model predicted incorrectly
roberta_correct_xlmroberta_incorrect = df_roberta[(df_roberta["GT"] == df_roberta["predicted"]) &
                                                  (df_xlmroberta["GT"] != df_xlmroberta["predicted"])]
roberta_tokenizer = TorchTokenizer("roberta-base")
xlmroberta_tokenizer = TorchTokenizer("xlm-roberta-base")

print_text_and_tokenization(roberta_correct_xlmroberta_incorrect, roberta_tokenizer, xlmroberta_tokenizer)

roberta_incorrect_xlmroberta_correct = df_roberta[(df_roberta["GT"] != df_roberta["predicted"]) &
                                                    (df_xlmroberta["GT"] == df_xlmroberta["predicted"])]
print_text_and_tokenization(roberta_incorrect_xlmroberta_correct, roberta_tokenizer, xlmroberta_tokenizer)
