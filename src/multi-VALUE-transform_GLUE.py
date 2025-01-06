"""
    script to transform GLUE tasks into multi-dialect tasks using multi-value (https://github.com/SALT-NLP/multi-value),
    meant to be run on the UU cluster
"""
import os
import gc

from styletokenizer.utility.env_variables import set_cache

set_cache()

from styletokenizer.utility.datasets_helper import load_data
from styletokenizer.glue import GLUE_TASKS, task_to_keys
from tqdm import tqdm
from styletokenizer.utility.custom_logger import log_and_flush
from multivalue import Dialects


class MultiDialect(Dialects.DialectFromVector):
    def __init__(self, **kwargs):
        super().__init__(dialect_name="all", **kwargs)


def main():
    """
        excerpt from multi-value paper:
            Multi-VALUE as a multidialectal augmentation tool by training on a synthetic pseudo-dialect
            that contains the union of all feature options (Multi)
        --> we transform GLUE to multi-VALUE GLUE that contains the union of all feature options
        POTENTIALLY, we could train on original GLUE and test on multi-VALUE GLUE to test robustness without fine-tuning

        This version uses batched map and explicit cleanup to help avoid OOM.
    """
    multidialect = MultiDialect()
    for task in GLUE_TASKS:
        log_and_flush(f"=== Processing task: {task} ===")
        org_task_data = load_data(task)
        text_fields = task_to_keys[task]

        # We’ll store error info for each task here
        task_error_info = []

        for split in tqdm(org_task_data.keys(), desc=f"Processing splits for {task}"):
            log_and_flush(f"Processing split: {split}")

            output_path = (
                f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/"
                f"{task}_multi/{split}.csv"
            )

            # Skip if file already exists
            if os.path.exists(output_path):
                log_and_flush(f"File {output_path} already exists. Skipping.")
                continue

            # Dictionary to keep track of errors for this split
            error_info = {"count": 0, "ids": []}

            # Define a batched transformation function
            def transform_batch(batch):
                new_batch = {}
                # We'll iterate over each column in the batch
                for key, value_list in batch.items():
                    if key in text_fields:
                        transformed_list = []
                        for i, text_val in enumerate(value_list):
                            try:
                                transformed_list.append(multidialect.transform(text_val))
                            except Exception:
                                transformed_list.append(text_val)  # fallback to original
                                error_info["count"] += 1
                                # If there's an 'idx' column, log the exact ID
                                if "idx" in batch:
                                    error_info["ids"].append(batch["idx"][i])
                        new_batch[key] = transformed_list
                    else:
                        # Keep non-text fields as is
                        new_batch[key] = value_list
                return new_batch

            # Apply the batched transformation
            # - batched=True and a moderate batch_size helps reduce memory peaks
            transformed_dataset = org_task_data[split].map(
                transform_batch,
                batched=True,
                batch_size=8,
                num_proc=1
            )

            # Save the transformed dataset to CSV
            transformed_dataset.to_csv(output_path)
            log_and_flush(f"Saved transformed dataset to {output_path}")

            # Log error information
            if error_info["count"] > 0:
                err_msg = (
                    f"Encountered {error_info['count']} transformation errors "
                    f"out of {len(transformed_dataset)} for split {split}. "
                    f"Problematic IDs: {error_info['ids']}"
                )
                log_and_flush(err_msg)
                task_error_info.append(err_msg)

            # Clean up references to this split’s data
            del transformed_dataset
            gc.collect()

        # Clean up after finishing this task
        del org_task_data
        gc.collect()

        # Optionally, log all errors for the task
        if task_error_info:
            log_and_flush(f"--- Error summary for {task} ---")
            for err in task_error_info:
                log_and_flush(err)


if __name__ == "__main__":
    main()