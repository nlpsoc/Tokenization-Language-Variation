#!/bin/sh

### Job name
#SBATCH --job-name=FIT-TOK

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/FIT-TOKENIZER_NO-PRETOKENIZER_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 48:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH -p gpu --gpus-per-node=2g.20gb:1
#SBATCH --mem 120G
#SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

python AV-transform-SNLI.py