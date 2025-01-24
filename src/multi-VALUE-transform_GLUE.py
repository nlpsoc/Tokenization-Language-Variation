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
import nltk
import random

from multivalue import Dialects
# dialectal variations used for synthetic variations in the multi-VALUE paper: https://aclanthology.org/2023.acl-long.44.pdf
#   - AppE (Appalachian English), in library as AppalachianDialect
#   - ChcE (Chicano English), in library as ChicanoDialect
#   - CollSgE (Colloquial Singapore English), in library as ColloquialSingaporeDialect
#   - IndE (Indian English), in library as IndianDialect
#   - UAAVE (Urban African American English), in library as AfricanAmericanVernacular
# Validated the following dialects in their transformation rules:
#   - Appalachian (4)
#   - Chicano (29 annotators)
#   - Indian (11)
#   - Urban African American (1)
#   - Colloquial American (13)
#   - Aboriginal (4)
#   - North of England(3)
#   - Ozark (3)
#   - Southeast American Enclave (3)
#   - Black South African English (1).
DIALECTS = {
    "AppE": Dialects.AppalachianDialect(),
    "ChcE": Dialects.ChicanoDialect(),
    "CollSgE": Dialects.ColloquialSingaporeDialect(),
    "IndE": Dialects.IndianDialect(),
    "UAA5VE": Dialects.AfricanAmericanVernacular(),
}

def approximate_sentence_count(text):
    # NLTK’s default tokenizer can handle many abbreviations better
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 1
    return len(sentences)


def main():
    """
        excerpt from multi-value paper:
            Multi-VALUE as a multidialectal augmentation tool by training on a synthetic pseudo-dialect
            that contains the union of all feature options (Multi)
        --> we transform GLUE to multi-VALUE GLUE that contains the union of all feature options
        POTENTIALLY, we could train on original GLUE and test on multi-VALUE GLUE to test robustness without fine-tuning

        This version uses batched map and explicit cleanup to help avoid OOM.
    """
    for task in ["mrpc", "rte", "wnli", "cola"]:  # GLUE_TASKS:
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
                # Prepare the new batch storage
                new_batch = {}

                # Initialize lists for each field to append per row
                for col in batch.keys():
                    new_batch[col] = []

                # Create a list to store the dialect used per row
                dialect_used_list = []

                # Number of rows in this batch
                n_rows = len(batch["idx"])

                # Iterate over each row i
                for i in range(n_rows):
                    # Shuffle dialects
                    dialect_order = list(DIALECTS.keys())
                    random.shuffle(dialect_order)

                    chosen_dialect = "SAE"  # default
                    dialect_found = False

                    # Store candidate transformed text for row i in a dict
                    candidate_transforms = {}

                    # Attempt to find a working dialect
                    for d_key in dialect_order:
                        transformation_failed = False
                        candidate_transforms[d_key] = {}

                        # For each column in the row
                        for key, value_list in batch.items():
                            # If it's not a text field, skip
                            if key not in text_fields:
                                continue

                            text_val = value_list[i]

                            # Define a ratio threshold. For example, 40 words per sentence is quite high.
                            num_words = len(text_val.split())
                            sent_count = approximate_sentence_count(text_val)
                            ratio_threshold = 40.0
                            avg_sentence_len = num_words / sent_count if sent_count else num_words

                            # Condition to SKIP transform: (multi-value seems to have trouble otherwise)
                            # - more than 100 words
                            # - average sentence length above our threshold
                            if (num_words > 150) and (avg_sentence_len > ratio_threshold):
                                # Keep original
                                candidate_transforms[d_key][key] = text_val
                                log_and_flush(
                                    f"Skipping transformation: row={i}, dialect={d_key}, "
                                    f"avg sent len={avg_sentence_len}, text length={num_words}, "
                                    f"id={batch['idx'][i]}, snippet={text_val[:200]}..."
                                )
                            else:
                                # Attempt transform with current dialect
                                try:
                                    transformed_text = DIALECTS[d_key].transform(text_val)
                                    if transformed_text.strip() != text_val.strip():  # transformation successful
                                        candidate_transforms[d_key][key] = transformed_text
                                    else:  # no meaningful transformation, basically failed
                                        transformation_failed = True
                                except Exception as e:
                                    error_info["count"] += 1
                                    if "idx" in batch:
                                        error_info["ids"].append(batch["idx"][i])

                                    log_and_flush(
                                        f"Error transforming text: row={i}, dialect={d_key}, "
                                        f"error={e}, snippet={text_val[:200]}..."
                                    )
                                    transformation_failed = True
                                    break

                        if not transformation_failed:
                            # Found a dialect that works for all text fields
                            chosen_dialect = d_key
                            dialect_found = True
                            break

                    # Save the final results for row i
                    # If we found a working dialect, use candidate_transforms[chosen_dialect]
                    # Otherwise, fallback to the original text for text fields
                    for key, value_list in batch.items():
                        if key in text_fields:
                            if dialect_found:
                                new_batch[key].append(candidate_transforms[chosen_dialect][key])
                            else:
                                # Fallback to original
                                new_batch[key].append(value_list[i])
                        else:
                            # Non-text field: just copy original
                            new_batch[key].append(value_list[i])

                    # Save the dialect used for this row
                    dialect_used_list.append(chosen_dialect)

                # Finally, store the dialect_used column as well
                new_batch["dialect_used"] = dialect_used_list

                return new_batch

            # Apply the batched transformation
            # - batched=True and a moderate batch_size helps reduce memory peaks
            transformed_dataset = org_task_data[split].map(
                transform_batch,
                batched=True,
                batch_size=1,
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
