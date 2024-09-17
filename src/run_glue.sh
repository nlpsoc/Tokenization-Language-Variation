#!/bin/sh

### Job name
#SBATCH --job-name=GLUE-3.3B-43

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/GLUE_3.3-43_%j.txt

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
export MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/3270M/steps-80000/seed-43
# export MODEL_PATH=/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-32000/329M/steps-80000/seed-44

# Extract the last part of the path
MODEL_NAME="${MODEL_PATH#*/tiny-BERT/}"

# #########################################################################################
#
export TASK_NAME=mrpc
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

export TASK_NAME=sst2
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

export TASK_NAME=qqp
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

export TASK_NAME=mnli
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

export TASK_NAME=qnli
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

export TASK_NAME=rte
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44

#export TASK_NAME=cola
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
#export TASK_NAME=wnli
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 5 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44
#
#export TASK_NAME=stsb
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/42/ --seed 42
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/43/ --seed 43
#python run_glue.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
#--max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 \
#--output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/bert-tiny/$MODEL_NAME/$TASK_NAME/44/ --seed 44