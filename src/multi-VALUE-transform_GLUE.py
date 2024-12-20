"""
    script to transform GLUE tasks into multi-dialect tasks using multi-value (https://github.com/SALT-NLP/multi-value),
    meant to be run on the UU cluster
"""
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
    """
    multidialect = MultiDialect()
    for task in GLUE_TASKS:
        org_task_data = load_data(task)
        text_fields = task_to_keys[task]

        for split in tqdm(org_task_data.keys(), desc="Processing splits"):
            log_and_flush(f"Processing {split}")
            # Dictionary to keep track of errors
            error_info = {"count": 0, "ids": []}

            output_path = (f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/{task}_multi/"
                           f"{split}.csv")
            # skip if file already exists
            if output_path.exists():
                log_and_flush(f"File {output_path} already exists. Skipping.")
                continue

            def transform_example(example):
                new_example = {}
                for key, value in example.items():
                    if key in text_fields:
                        try:
                            new_example[key] = multidialect.transform(value)
                        except Exception as e:
                            # Transformation failed; revert to original and log the error
                            new_example[key] = value
                            error_info["count"] += 1
                            # Assuming there's an 'id' field or something similar
                            # If not, you may store indices or some other identifier
                            if "idx" in example:
                                error_info["ids"].append(example["idx"])
                    else:
                        new_example[key] = value
                return new_example

            # Use dataset.map to apply transformations to specified text fields
            transformed_dataset = org_task_data[split].map(
                transform_example,
                batched=False,
                num_proc=1
            )
            # Save the transformed dataset to CSV

            transformed_dataset.to_csv(output_path)
            log_and_flush(f"Saved transformed dataset to {output_path}")
            # Log error information
            if error_info["count"] > 0:
                log_and_flush(
                    f"Encountered {error_info['count']} transformation errors out of {len(transformed_dataset)}. "
                    f"Problematic IDs: {error_info['ids']}"
                )


if __name__ == "__main__":
    main()
