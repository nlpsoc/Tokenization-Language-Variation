#!/bin/sh

### Job name
#SBATCH --job-name=TRAIN-3.3B

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/PRETRAIN_bert-wikibook_330M_44_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 30:00:00

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

# 1600M words
# python train_bert.py --uu --seed 42 --word_count 1_600_000_000 --epochs 6 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# python train_bert.py --uu --seed 42 --word_count 1_600_000_000 --epochs 6 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# python train_bert.py --uu --seed 42 --word_count 1_600_000_000 --epochs 6 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000

# 330M words
# python train_bert.py --uu --seed 42 --word_count 330_000_000 --epochs 10 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
# python train_bert.py --uu --seed 43 --word_count 330_000_000 --epochs 10 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000
python train_bert.py --uu --seed 44 --word_count 330_000_000 --epochs 10 --tokenizer /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/mixed-gpt2-32000