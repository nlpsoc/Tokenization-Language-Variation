#!/bin/sh

### Job name
#SBATCH --job-name=RB-43_llama3

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o OUTPUT_PATH.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 48:00:00

### Request a specific GPU
#SBATCH -p gpu --gpus-per-node=2g.20gb:1

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G
#SBARCH --gres=tmpspace:100G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

MODEL_PATHS=(
#  "PATH-TO-MODFEL"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-43"
  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-ws-32000/749M/steps-45000/seed-43"
)

SEEDS=(42
# 43
# 44
)

OUTPUT_DIR="/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output"

#Train/dev file paths are currently fixed here

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="${MODEL_PATH#*/models/}"
  echo "MODEL_NAME: $MODEL_NAME"

  # Extract the last part of the path
  MODEL_NAME="GLUE/${MODEL_PATH#*/models/}"
  PER_DEVICE_TRAIN_BATCH_SIZE=32
  MAX_SEQ_LENGTH=128

  for SEED in "${SEEDS[@]}"; do
    echo "SEED: $SEED"
    # ========= GLUE =========

    export TASK_NAME=sst2
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch

    export TASK_NAME=qqp
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch

    export TASK_NAME=mnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch
    #
    export TASK_NAME=qnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch

    # ===== TEXTFLINT-TRANSFORMED GLUE =====

    MODEL_NAME="GLUE/textflint/${MODEL_PATH#*/models/}"

    export TASK_NAME=sst2
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/SST2/sst2_train_textflint.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/SST2/sst2_dev_textflint.csv

    export TASK_NAME=qqp
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QQP/qqp_train_textflint.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QQP/qqp_val_textflint.csv
#
    export TASK_NAME=mnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/MNLI/mnli_train_textflint.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/MNLI/mnli_val_matched_textflint.csv,/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/MNLI/mnli_val_mismatched_textflint.csv

    export TASK_NAME=qnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QNLI/qnli_train_textflint.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/GLUE_textflint/QNLI/qnli_val_textflint.csv

    # ==== multi-VALUE Transformation ====

    MODEL_NAME="GLUE/mVALUE/${MODEL_PATH#*/models/}"

    export TASK_NAME=sst2
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/sst2_multi/train.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/sst2_multi/validation.csv

    export TASK_NAME=qqp
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/qqp_multi/train.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/qqp_multi/validation.csv
#
    export TASK_NAME=mnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/mnli_multi/train.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/mnli_multi/validation_matched.csv,/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/mnli_multi/validation_mismatched.csv

    export TASK_NAME=qnli
    python run_glue_org.py --model_name_or_path $MODEL_PATH --task_name $TASK_NAME --do_train --do_eval \
    --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --learning_rate 2e-5 --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR/$MODEL_NAME/$SEED/$TASK_NAME --seed $SEED --overwrite_cache --save_strategy epoch \
    --train_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/qnli_multi/train.csv \
    --validation_file /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/qnli_multi/validation.csv

  done

done
