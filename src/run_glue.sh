#!/bin/sh

### Job name
#SBATCH --job-name=GLUE-3.3B-42

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/GLUE_base-750M-45k-42_mixed-gpt2-32k_42_%j.txt

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
# export MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-80000/seed-43
# export MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/329M/steps-80000/seed-44
# -----
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-llama3-32000/3270M/steps-75000/seed-42
# MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-ws-32000/3270M/steps-75000/seed-42
MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-no-32000/3270M/steps-75000/seed-42
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

# Extract the last part of the path
MODEL_NAME="GLUE/${MODEL_PATH#*/models/}"
PER_DEVICE_TRAIN_BATCH_SIZE=32
MAX_SEQ_LENGTH=128

## #########################################################################################
##  add --do_predict to eval on test set
##export TASK_NAME=mrpc
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#


export TASK_NAME=sst2
python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy no
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
export TASK_NAME=qqp
python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy no
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
export TASK_NAME=mnli
python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy no
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
export TASK_NAME=qnli
python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy no
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#



##export TASK_NAME=rte
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
##export TASK_NAME=cola
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
##
##export TASK_NAME=wnli
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
##
##export TASK_NAME=stsb
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
##python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
##--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
##--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44