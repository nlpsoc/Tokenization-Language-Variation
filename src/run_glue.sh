#!/bin/sh

### Job name
#SBATCH --job-name=GLUE-3.3B-42

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/GLUE-textflint_base-750M-45k_42_%j.txt

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

MODEL_PATHS=(
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-no-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-ws-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/twitter-gpt2-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/pubmed-gpt2-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/webbook-gpt2-32000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-500/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-128000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-4000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-16000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-64000/749M/steps-45000/seed-42"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-43"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-44"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/tiny-BERT/mixed-gpt2-128000/3270M/steps-300000/seed-42"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="${MODEL_PATH#*/models/}"
  echo "MODEL_NAME: $MODEL_NAME"

  # Extract the last part of the path
  MODEL_NAME="GLUE/${MODEL_PATH#*/models/}"
  PER_DEVICE_TRAIN_BATCH_SIZE=32
  MAX_SEQ_LENGTH=128

  # ========= GLUE =========

  #  export TASK_NAME=sst2
  #  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  #  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  #  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch
  #  #
  #  export TASK_NAME=qqp
  #  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  #  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  #  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch
  #  #
  #  export TASK_NAME=mnli
  #  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  #  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  #  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch
  #  #
  #  export TASK_NAME=qnli
  #  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  #  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  #  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch

  # ===== TEXTFLINT-TRANSFORMED GLUE =====

  MODEL_NAME="GLUE/textflint/${MODEL_PATH#*/models/}"

  export TASK_NAME=sst2
  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch \
  --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/SST2/sst2_train_textflint.csv \
  --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/SST2/sst2_dev_textflint.csv

  export TASK_NAME=qqp
  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch \
  --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QQP/qqp_train_textflint.csv \
  --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QQP/qqp_val_textflint.csv

  export TASK_NAME=mnli
  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch \
  --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/MNLI/mnli_train_textflint.csv \
  --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/MNLI/mnli_val_matched_textflint.csv

  export TASK_NAME=qnli
  python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
  --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/$MODEL_NAME/42/$TASK_NAME --seed 42 --overwrite_cache --save_strategy epoch \
  --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QNLI/qnli_train_textflint.csv \
  --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QNLI/qnli_val_textflint.csv

done
