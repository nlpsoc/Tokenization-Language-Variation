"""
    calculate the correlations between BERT predictions and log regression / intrinsic measures
"""
import json
import os

from styletokenizer.glue import GLUE_TEXTFLINT_TASKS
from styletokenizer.tokenizer import TOKENIZER_PATHS

performance_keys = {
    "sst2": "eval_accuracy",
    "qqp": "eval_f1",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
}


def main():
    # do this only for the textflint tasks for now
    tasks = GLUE_TEXTFLINT_TASKS
    tokenizer_paths = TOKENIZER_PATHS

    unique_tokenizer_paths = set()
    for tokenizer_path in tokenizer_paths:
        for path in tokenizer_path:
            unique_tokenizer_paths.add(path)

    # collect the BERT performance scores of the tasks
    BERT_PERFORMANCE = {}
    local_finder_addition = "/Users/anna/sftp_mount/hpc_disk/02-awegmann/"
    server_finder_addition = "/hpc/uu_cs_nlpsoc/02-awegmann/"
    GLUE_OUT_BASE_PATH = os.path.join(server_finder_addition, "TOKENIZER/output/GLUE/textflint/base-BERT/")
    BERT_PATH = "749M/steps-45000/seed-42/42/"

    for task in tasks:
        task_key = task
        if task in GLUE_TEXTFLINT_TASKS:
            task_key = task.split('-')[0]

        for tokenizer_path in unique_tokenizer_paths:
            # get tokenizer name
            tokenizer_name = os.path.basename(os.path.dirname(tokenizer_path))
            # get the BERT output for the task
            result_path = os.path.join(GLUE_OUT_BASE_PATH, tokenizer_name, BERT_PATH, task_key)
            # check that path exists
            if not os.path.exists(result_path):
                print(f"Path {result_path} does not exist")
                continue

            # read in json file
            with open(result_path, "r") as f:
                data_dict = json.load(f)
            # get the performance from the performance keys
            BERT_PERFORMANCE[task_key] = data_dict[performance_keys[task_key]]

    print(BERT_PERFORMANCE)


if __name__ == "__main__":
    main()
