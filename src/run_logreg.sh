#!/bin/sh

### Job name
#SBATCH --job-name=LR-EVAL

### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
# ------------------ SENTENCE BERTS -----------------------------------
#SBATCH -o /hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/output/LOGREG_mVALUE_%j.txt

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute:second,
#SBATCH -t 48:00:00

### Memory your job needs per node, e. g. 1 GB
#SBATCH --mem 120G

source /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/local/Rocky8/uu_cs_nlpsoc/miniconda3/envs/aw_tokenizer

# python run_logreg.py --task "mnli-textflint,qnli-textflint"  --tokenizer_paths "all"  # qqp-textflint,sst2-textflint,mnli-textflint,qnli-textflint
python run_logreg.py --task "qqp-mVALUE,sst2-mVALUE,qnli-mVALUE,mnli-mVALUE"  --tokenizer_paths "all"
# python run_logreg.py --task "sadiri"  --tokenizer_paths "all" --on_test_set
# python run_logreg.py --task "NUCLE"  --tokenizer_paths "all"
# --tokenizer_paths "/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/pubmed-gpt2-32000/tokenizer.json,/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/tokenizer/webbook-gpt2-32000/tokenizer.json"