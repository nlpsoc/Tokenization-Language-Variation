import os
import pandas as pd
import torch
import transformers
import numpy as np
import argparse
import re

from torch import nn
import torch.nn.functional as F
from transformers import AdamW, RobertaTokenizerFast, BertTokenizerFast
from nltk import word_tokenize
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from styletokenizer.utility.env_variables import set_torch_device
from styletokenizer.utility.filesystem import get_dir_to_src


class RNN(nn.Module):

    def __init__(self, hidden_dim, vocab, out_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, num_layers=2, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, out_dim))

    def forward(self, texts, lens):
        out = self.embedding(texts)
        packed_input = pack_padded_sequence(out, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        #        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
        out = self.linear(out)
        return out

    def predict(self, texts):
        out = self.embedding(texts)
        packed_output, (ht, ct) = self.lstm(out)
        #        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
        out = self.linear(out)
        return out
