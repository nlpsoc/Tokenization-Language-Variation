from dataclasses import dataclass
from datasets import Dataset
from typing import Dict, List, Optional, Union

import torch
import pandas as pd
from transformers import AutoTokenizer
from styletokenizer.sadiri.mask import masking, load_top_tokens
from tqdm import tqdm
from styletokenizer.sadiri.cluster_batches import ClusterData

@dataclass
class AATrainData:
    dataloader: Dataset = None
    clustering: ClusterData = None
    batch_size: int = 64
    
    def __call__(self, dataloader):
        self.dataloader = dataloader
        
    def __len__(self):
        return len(self.dataloader)
    
    def on_epoch(self):
        if self.clustering.empty():
            print("applying random clustering")
            df = self.dataloader.to_pandas()
            column_labels = list(df.columns)
            new_df = pd.DataFrame(columns=column_labels)
            author_ids = self.clustering.cluster_random(len(df))
            for idx in tqdm(author_ids):
                row = df.iloc[idx]
                new_df = pd.concat([new_df, row.to_frame().T])
            new_df.reset_index(drop=True, inplace=True)
            del df
            self.dataloader = Dataset.from_pandas(new_df)
            del new_df
            print("finished!")
        else:
            print("applying kmean clustering")
            df = self.dataloader.to_pandas()
            column_labels = list(df.columns)
            new_df = pd.DataFrame(columns=column_labels)
            cluster_author_ids = self.clustering.cluster_hard_negative()
            author_ids = []
            for cluster in cluster_author_ids:
                author_ids += cluster
            for idx in tqdm(author_ids):
                row = df.iloc[idx]
                new_df = pd.concat([new_df, row.to_frame().T])
            new_df.reset_index(drop=True, inplace=True)
            del df
            self.dataloader = Dataset.from_pandas(new_df)
            del new_df
            print("finished!")
    
    
@dataclass
class TextDataCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 350
    evaluate: bool = False
    mask: float = 0.8
    path: str = ''
    top: int = 200
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        
        # No masking
        if self.mask == 0.0:
            sentA = [{'input_ids': self.tokenizer(feature['query_text'])[
                'input_ids'][:self.max_length]} for feature in features]
            sentB = [{'input_ids': self.tokenizer(feature['candidate_text'])[
                'input_ids'][:self.max_length]} for feature in features]
            
        # mask top words
        elif self.mask <= 1.0:
            top_tokens = load_top_tokens(self.path, self.tokenizer, top=self.top)
            special = self.tokenizer.encode("test")
            top_tokens.append(special[0])
            top_tokens.append(special[-1])
            top_tokens = set(top_tokens)
            mask_token = self.tokenizer.mask_token_id
            sentA = [{'input_ids': masking(
                top_tokens = top_tokens,
                mask_token = mask_token,
                rate = self.mask,
                sentence = self.tokenizer(feature['query_text'])['input_ids'][:self.max_length]
                )} for feature in features]
            sentB = [{'input_ids': masking(
                top_tokens = top_tokens,
                mask_token = mask_token,
                rate = self.mask,
                sentence = self.tokenizer(feature['candidate_text'])['input_ids'][:self.max_length]
                )} for feature in features]

        batchA = self.tokenizer.pad(
            sentA,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        
        batchB = self.tokenizer.pad(
            sentB,
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        
        
        if self.evaluate:
            query_authors = [feature['query_authorID'] for feature in features]
            target_authors = [feature['candidate_authorID']
                              for feature in features]

            return batchA, batchB, query_authors, target_authors
        else:
            return batchA, batchB
