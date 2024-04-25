#!/bin/bash

source /opt/anaconda/bin/activate

CUDA_VISIBLE_DEVICES=6 python3 lstm_jian.py > ../output/24-04-25_reproduce-bert-uncased_out.txt 2>&1