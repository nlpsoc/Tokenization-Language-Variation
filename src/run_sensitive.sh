#!/bin/sh

### Job name
#SBATCH --job-name=S_44_wiki

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/SENSITIVE_44_wiki_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 30:00:00

### Request a specific GPU
### see https://hpcusers.op.umcutrecht.nl/xwiki/bin/view/Main/Setup/Cluster/GPU-nodes/
### --> 2g20gb is 1/3 of a A100 GPU
### --> 7g.79gb is A100 GPU (4 in the whole cluster)
#SBATCH -p gpu --gpus-per-node=2g.20gb:1
### SBATCH -p gpu --gpus-per-node=7g.79gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem-per-gpu 79G
### SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer


MODEL_PATHS=(
#  paths to models to evaluate
"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-44"
)

SEEDS=(42
# 43
# 44
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="${MODEL_PATH#*/models/}"
  echo "MODEL_NAME: $MODEL_NAME"

  for SEED in "${SEEDS[@]}"; do
    echo "SEED: $SEED"

    TASK_NAME=PAN
    python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
    --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

    TASK_NAME=multi-DIALECT
    python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
    --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

    TASK_NAME=sadiri
    python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
    --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

    TASK_NAME=NUCLE
    python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
    --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

    TASK_NAME=CORE
    python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
    --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME


  done

done


