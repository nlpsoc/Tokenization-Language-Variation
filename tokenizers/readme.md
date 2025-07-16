Note that due to the original script used to fit tokenizers, they only include bytes that occurred in the pre-training corpus. Others are mapped to `[UNK]`.

This should not lead to drastically different results compared to performing all experiments with a tokenizer starting out with all 256 bytes.  
This is because fitting corpora are all big and downstream corpora should only include bytes that are also present in the training/fitting corpus (the misc corpus). 

This documentation includes the tokenizers + the evaluation scores.