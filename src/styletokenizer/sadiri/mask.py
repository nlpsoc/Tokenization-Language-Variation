import random
import os
import json
from tqdm import tqdm
from transformers import RobertaTokenizer
from collections import Counter


def masking(top_tokens, mask_token, rate, sentence):
    random_values = [random.random() for _ in sentence]
    sent = [mask_token if word not in top_tokens and rand_val < rate else word for word, rand_val in zip(sentence, random_values)]
    # return ' '.join(sent)
    return sent

def load_top_tokens(path, tokenizer, top):
    fullpath = os.path.join(path, 'top.txt')
    if os.path.exists(fullpath):
        with open(fullpath, 'r') as file:
            top_token_ids = file.read().splitlines()[0:top]
    else:
        print("generating top tokens from dataset...")
        token_count = Counter()
        with open(os.path.join(path, 'train.jsonl'), 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)
                input_ids = tokenizer.encode(
                    line['query_text']
                    )[1:-1]
                token_count.update(input_ids)
                
                input_ids2 = tokenizer.encode(
                    line['candidate_text']
                    )[1:-1]
                token_count.update(input_ids2)
                
        top_frequent = token_count.most_common(1000)
        top_token_ids = [tup[0] for tup in top_frequent]
        #save the top tokens
        with open(fullpath, 'w') as file:
            file.writelines([str(item) + '\n' for item in top_token_ids])
        print("Top token saved! ")
        top_token_ids = top_token_ids[0:top]
    return [int(s) for s in top_token_ids]

if __name__ == '__main__':
    path = '/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_1.2'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    # load_top_tokens(path, tokenizer, 1000)
    
    load_top_tokens('/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_1.2', tokenizer, 1000)
    load_top_tokens('/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_1.3', tokenizer, 1000)
    load_top_tokens('/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_1.4', tokenizer, 1000)
    load_top_tokens('/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_1.5', tokenizer, 1000)
    load_top_tokens('/shared/3/projects/hiatus/sadiri-mask/hrs_datasets/hrs_crossGenre', tokenizer, 1000)
    
    