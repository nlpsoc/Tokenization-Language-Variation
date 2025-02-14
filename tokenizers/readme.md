Note that tokenizers only include bytes that occurred in the pre-training corpus. Others are mapped to `[UNK]`. 
This should not lead to different results compared to performing all experiments with a tokenizer starting out with all 256 bytes. 
This is because fitting corpora are big and diverse and downstream corpora should only include bytes that are also present in the misc fitting corpus. 