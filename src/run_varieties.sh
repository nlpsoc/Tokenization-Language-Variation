#!/bin/sh

### Job name
#SBATCH --job-name=VAR-0.3B

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR_0.3-42_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 24:00:00

### Request a specific GPU
### see https://hpcusers.op.umcutrecht.nl/xwiki/bin/view/Main/Setup/Cluster/GPU-nodes/
### --> 2g20gb is 1/3 of a A100 GPU
### --> 7g.79gb is A100 GPU (4 in the whole cluster)
#SBATCH -p gpu --gpus-per-node=2g.20gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G
#SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

# export MODEL_PATH=prajjwal1/bert-tiny
MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/329M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3247M/steps-75000/seed-42
SEED=42
MODEL_NAME="${MODEL_PATH#*/tiny-BERT/}"
TASK_NAME=sadiri
out_put_root=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED

python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir $out_put_root


