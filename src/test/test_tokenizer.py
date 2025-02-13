from unittest import TestCase
import os
from train_bert import load_tokenizer


class TestTokenizer(TestCase):

    def test_tokenizers(self):

        local_finder_addition = "/Users/anna/sftp_mount/hpc_disk2/02-awegmann/"
        no_tok, _ = load_tokenizer(tokenizer_no)
        wsorg_tok, _ = load_tokenizer(tokenizer_wsorg)
        ws_tok, _ = load_tokenizer(tokenizer_ws)
        gpt2_tok, _ = load_tokenizer(tokenizer_gpt2)
        llama3_tok, _ = load_tokenizer(tokenizer_llama3)
        tok_500, _ = load_tokenizer(tokenizer_500)
        tok_4k, _ = load_tokenizer(tokenizer_4k)
        tok_64k, _ = load_tokenizer(tokenizer_64k)
        tok_128k, _ = load_tokenizer(tokenier_128k)
        tok_twitter, _ = load_tokenizer(tokenizer_twitter)
        tok_pubmed, _ = load_tokenizer(tokenizer_pubmed)
        tok_wikipedia, _ = load_tokenizer(tokenizer_wikipedia)

        TOKENIZERS = {"gpt2": gpt2_tok, "llama3": llama3_tok, "wsorg": wsorg_tok, "ws": ws_tok, "no": no_tok,
                      "500": tok_500, "4k": tok_4k, "64k": tok_64k, "128k": tok_128k, "twitter": tok_twitter,
                      "pubmed": tok_pubmed, "wikipedia": tok_wikipedia}

