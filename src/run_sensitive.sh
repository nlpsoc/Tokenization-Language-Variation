#!/bin/sh

### Job name
#SBATCH --job-name=S_43_wiki

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/SENSITIVE_43_wiki_%j.txt

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
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-no-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-wsorg-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-ws-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/twitter-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/pubmed-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-500/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-128000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-4000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-64000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/base-BERT/webbook-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-no-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-wsorg-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-ws-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/twitter-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/pubmed-gpt2-32000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-500/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-4000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-64000/749M/steps-45000/seed-42"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-128000/749M/steps-45000/seed-42"
###
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-128000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-500/749M/steps-45000/seed-43"
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-64000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-4000/749M/steps-45000/seed-43"
"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-43"
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/pubmed-gpt2-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/twitter-gpt2-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-wsorg-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-no-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-43"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-ws-32000/749M/steps-45000/seed-43"
###
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-128000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-500/749M/steps-45000/seed-44"
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-64000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-4000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/wikipedia-gpt2-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-gpt2-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/pubmed-gpt2-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/twitter-gpt2-32000/749M/steps-45000/seed-44"
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-wsorg-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-no-32000/749M/steps-45000/seed-44"
# "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-llama3-32000/749M/steps-45000/seed-44"
#  "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/models/train-mixed/base-BERT/mixed-ws-32000/749M/steps-45000/seed-44"
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



#  TASK_NAME=convo-style
#  SEED=42
#  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

#  TASK_NAME=age
#  SEED=42
#  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME




#  TASK_NAME=GYAFC
#  SEED=42
#  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME
#  #
#  TASK_NAME=DIALECT
#  SEED=42
#  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$MODEL_NAME/$SEED/$TASK_NAME

#  TASK_NAME=CGLU
#  SEED=42
#  python run_varieties.py --model_path $MODEL_PATH --task $TASK_NAME --seed $SEED \
#  --output_dir /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/VAR/$TASK_NAME/$MODEL_NAME/$SEED

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

