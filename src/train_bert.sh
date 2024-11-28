#!/bin/sh

### Job name
#SBATCH --job-name=3.3-75k-BERT

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/PRETRAIN_BASE_750M-45k-42_mixed-gpt2-32k_batch-32_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 48:00:00

### Request a specific GPU
### see https://hpcusers.op.umcutrecht.nl/xwiki/bin/view/Main/Setup/Cluster/GPU-nodes/
### --> 2g20gb is 1/3 of a A100 GPU
### --> 7g.79gb is A100 GPU (4 in the whole cluster)
### SBATCH -p gpu --gpus-per-node=2g.20gb:4
#SBATCH -p gpu --gpus-per-node=7g.79gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G
#SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

# seeds, 3300M words, 80k
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 80000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# python train_bert.py --uu --seed 43 --word_count 3_300_000_000 --steps 80000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# python train_bert.py --uu --seed 44 --word_count 3_300_000_000 --steps 80000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# pre-tokenizers
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-llama3-32000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-ws-32000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# corpus
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/twitter-gpt2-32000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/wikipedia-gpt2-32000

# vocab size
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-500
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-1000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-2000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-4000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-8000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-16000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 150000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-64000 --batch_size 128
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 300000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-128000 --batch_size 64
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-256000
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-512000

# mini BERT
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/twitter-gpt2-32000 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/wikipedia-gpt2-32000 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-ws-32000 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-llama3-32000 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 75000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-500 --model_size 11
# python train_bert.py --uu --seed 42 --word_count 3_300_000_000 --steps 300000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-128000 --model_size 11 --batch_size 64


# medium BERT on 100 million words
# python train_bert.py --uu --seed 42 --word_count 100_000_000 --model_size 42 --batch_size 32 --steps 6000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# tiny BERT on 10 million words
# python train_bert.py --uu --seed 42 --word_count 10_000_000 --model_size 4 --batch_size 32 --steps 600 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# base BERT on 250 million words
# python train_bert.py --uu --seed 42 --word_count 250_000_000 --model_size 110 --batch_size 32 --steps 15_000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# base BERT on 750 million words (equating to about 12 hour of training time)
python train_bert.py --uu --seed 42 --word_count 750_000_000 --model_size 110 --batch_size 128 --steps 11_250 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# large BERT on 1 billion words
# python train_bert.py --uu --seed 42 --word_count 1_000_000_000 --model_size 336 --batch_size 32 --steps 61_000 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
