## Repo for Tokenization Is Sensitive To Language Variation

This repository contains code, artifacts and links to datasets for the paper ["Tokenization Is Sensitive To Language Variation."](https://arxiv.org/pdf/2502.15343) 

### Code

The `run_varieties.py`, `run_glue.py` are the fine-tuning of BERT models  
`train_bert.py` is training BERT  
`run_sensitive.sh` is a shell script demonstrating the evaluation of trained BERT on variation-sensitive tasks  
`run_robust.sh` is a shell script demonstrating the evaluation of trained BERT on variation-robust tasks 
`fit_tokenizer.py` is fitting the tokenizers  
`run_logreg.py` is the logistic regression model  
`intrinsic_eval.py` is the "intrinsic" evaluation of the tokenization models  

Planning to eventually release a slimmed down and refactored version of the code, such that it can be used to easily evaluate your own tokenizers. This could take some time, unfortunately.

### Datasets

Most scripts are sweat and tears relating to dataset creation (see `create_XXX` files). Unless you want to understand that process, ignore those files. 
Find the datasets ready to use on huggingface (except for those where we lack licenses to share):

**Fitting Datasets**
- Wikipedia: 
- PubMed: https://huggingface.co/datasets/AnnaWegmann/Fitting-PubMed
- Twitter: cannot be shared due to licensing constraints

**Training Datasets**
- Miscellaneous: 

**Tasks**  

*Variation-Sensitive*
- Authorship Verification:
- PAN:
- CORE:
- NUCLE: cannot be shared due to licensing constraints 
- Dialect: 

*Variation-Robust*
- GLUE: 
- GLUE+typo: 
- GLUE+dialect: 

## Requirements

- Python 3.11.9  

`requirements.txt`