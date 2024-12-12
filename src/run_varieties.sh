#!/bin/sh

### Job name
#SBATCH --job-name=VAR-3.3-42

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
### SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR_mini-3.3-75k-42_mixed-gpt2-500_42_%j.txt
### SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR_base-42_mixed-gpt2-32k_%j.txt
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/SYNTH_tiny-42_Style-NLI_MIX-MODELS_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 24:00:00

### Request a specific GPU
### see https://hpcusers.op.umcutrecht.nl/xwiki/bin/view/Main/Setup/Cluster/GPU-nodes/
### --> 2g20gb is 1/3 of a A100 GPU
### --> 7g.79gb is A100 GPU (4 in the whole cluster)
#SBATCH -p gpu --gpus-per-node=2g.20gb:1
### SBATCH -p gpu --gpus-per-node=7g.79gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G
#SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

# export MODEL_PATH=prajjwal1/bert-tiny
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-llama3-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-ws-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/twitter-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/wikipedia-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-500/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-1000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-2000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-4000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-8000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-16000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-64000/3270M/steps-150000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-128000/3270M/steps-300000/seed-42
# -----
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/mixed-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/twitter-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/wikipedia-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/mixed-ws-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/mixed-llama3-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/mini-BERT/mixed-gpt2-500/3270M/steps-75000/seed-42
# -----
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-42

MODEL_PATHS=(
  # "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-500/3270M/steps-75000/seed-42"
  # "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-75000/seed-42"
  # "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-ws-32000/3270M/steps-75000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-llama3-32000/3270M/steps-75000/seed-42"
  # "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/wikipedia-gpt2-32000/3270M/steps-75000/seed-42"
  # "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/twitter-gpt2-32000/3270M/steps-75000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-128000/3270M/steps-300000/seed-42"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="${MODEL_PATH#*/models/}"
  echo "MODEL_NAME: $MODEL_NAME"

  TASK_NAME=SNLI-NLI
  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME
  TASK_NAME=SNLI-Style
  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME
    TASK_NAME=SNLI
  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME

done

##TASK_NAME=stel
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME
##
##TASK_NAME=sadiri
##SEED=42
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#TASK_NAME=age
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#TASK_NAME=CORE
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#
#TASK_NAME=CGLU
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#
#TASK_NAME=GYAFC
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#
#TASK_NAME=DIALECT
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#
#
##TASK_NAME=mrpc
##SEED=42
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=43
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=44
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
## -----
#TASK_NAME=sst2
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
##SEED=43
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=44
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
## -----
#TASK_NAME=qqp
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
##SEED=43
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=44
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
## -----
#TASK_NAME=mnli
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
##SEED=43
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=44
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
## -----
#TASK_NAME=qnli
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
##SEED=43
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
##SEED=44
##python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
##--seed $SEED
# -----
#TASK_NAME=rte
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
#SEED=43
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
#SEED=44
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED



# CREATING the VALUE files
#MAPPING_FILE='/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/code/value/resources/sae_aave_mapping_dict.pkl'
#conda deactivate
#conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_value_old
#cd /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/code/value
#export VALUE_TASK_NAME=sst2
#python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
#--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
#export VALUE_TASK_NAME=qqp
#python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
#--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
#export VALUE_TASK_NAME=mnli
#python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
#--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
#export VALUE_TASK_NAME=qnli
#python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
#--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
#export VALUE_TASK_NAME=rte
#python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
#--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME

