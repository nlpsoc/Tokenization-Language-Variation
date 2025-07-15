## Repo for Tokenization Is Sensitive To Language Variation

This repository contains the code for the paper "Tokenization Is Sensitive To Language Variation."  
Most scripts are sweat and tears relating to dataset creation.   
The `run_varieties.py`, `run_glue.py` are the fine-tuning of BERT models  
`train_bert.py` is training BERT  
`run_sensitive.sh` is a shell script demonstrating the evaluation of trained BERT on variation-sensitive tasks  
`run_robust.sh` is a shell script demonstrating the evaluation of trained BERT on variation-robust tasks 
`fit_tokenizer.py` is fitting the tokenizers
`run_logreg.py` is the logistic regression model  
`intrinsic_eval.py` is the "intrinsic" evaluation of the tokenization models  

Unfortunately, the datasets are too big to share here. We will release them pending licensing issues.  
Planning to eventually release a slimmed down and refactored version of the code, such that it can be used to easily evaluate your own tokenizers.

## Requirements

- Python 3.11.9  

`requirements.txt`