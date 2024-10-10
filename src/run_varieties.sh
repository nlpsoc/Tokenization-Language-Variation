#!/bin/sh

### Job name
#SBATCH --job-name=VAR-3.3-42

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
### SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR_3.3-75k-42_mixed-gpt2-32k_VALUE-mrpc_42_%j.txt
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE_TRANSFORM_SST2-RTE_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 48:00:00

### Request a specific GPU
### see https://hpcusers.op.umcutrecht.nl/xwiki/bin/view/Main/Setup/Cluster/GPU-nodes/
### --> 2g20gb is 1/3 of a A100 GPU
### --> 7g.79gb is A100 GPU (4 in the whole cluster)
### SBATCH -p gpu --gpus-per-node=2g.20gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G
### SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

# export MODEL_PATH=prajjwal1/bert-tiny
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/_mixed-gpt2-32000/329M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/_mixed-gpt2-32000/3247M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-80000/seed-44
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/329M/steps-80000/seed-44
# -----
MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-75000/seed-42
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

MODEL_NAME="${MODEL_PATH#*/tiny-BERT/}"


#TASK_NAME=stel
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME
###
#TASK_NAME=sadiri
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
##SEED=43
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
##SEED=44
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
##
#TASK_NAME=age
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
##SEED=43
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
##SEED=44
##python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED

#TASK_NAME=CORE
#SEED=42
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#SEED=43
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED
#SEED=44
#python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED


# CREATING the VALUE files
MAPPING_FILE='/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/code/value/resources/sae_aave_mapping_dict.pkl'
conda deactivate
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_value_old
cd /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/code/value
export VALUE_TASK_NAME=sst2
python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
export VALUE_TASK_NAME=qqp
python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
export VALUE_TASK_NAME=mnli
python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
export VALUE_TASK_NAME=qnli
python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME
export VALUE_TASK_NAME=rte
python run_transform_glue.py --task_name $VALUE_TASK_NAME --dialect "aave" --lexical_mapping $MAPPING_FILE  \
--morphosyntax --model_name_or_path "prajjwal1/bert-tiny" --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VALUE/$TASK_NAME

#TASK_NAME=mrpc
#SEED=42
#python  run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/value/$TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#VALUE_TASK_NAME=sst2
#SEED=42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#VALUE_TASK_NAME=qqp
#SEED=42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#VALUE_TASK_NAME=mnli
#SEED=42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#VALUE_TASK_NAME=qnli
#SEED=42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#VALUE_TASK_NAME=rte
#SEED=42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
#--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=43
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
###SEED=44
###python run_glue.py --model_name_or_path $MODEL_PATH --task_name $VALUE_TASK_NAME --do_train --do_eval \
###--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
###--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$VALUE_TASK_NAME/$MODEL_NAME/$SEED \
###--seed $SEED  --dialect "aave" --lexical_mapping $MAPPING_FILE --morphosyntax
#
