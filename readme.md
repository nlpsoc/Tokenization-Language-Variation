## Repo for Tokenization Is Sensitive To Language Variation

This repository contains the code for the paper "Tokenization Is Sensitive To Language Variation."
Most scripts are sweat and tears relating to dataset creation. 
The run_varieties, run_glue are the fine-tuning of BERT models
train_bert is training BERT
fit_tokenizer is training the tokenization models
run_logreg is the logistic regression model
intrinsic_eval is the intrinsic evaluation of the tokenization models

Unfortunately, the datasets are too big to share here or on ARR. We will release them pending licensing issues.

## Requirements

- Python 3.11.9
see `requirements.txt`