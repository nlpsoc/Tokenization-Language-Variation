"""
    Generated with ChatGPT o1-preview on 2024-11-07
"""
import os
import random
import pandas as pd
from datasets import load_dataset, load_from_disk, concatenate_datasets


def get_text(example, task):
    if task == 'sst2':
        return example['sentence']
    elif task == 'qqp':
        return example['question1'] + " " + example['question2']
    elif task == 'mnli':
        return example['premise'] + " " + example['hypothesis']
    elif task == 'qnli':
        return example['question'] + " " + example['sentence']
    else:
        return ''


def create_datasets():
    # set seed
    random.seed(42)
    tasks = ['sst2', 'qqp', 'mnli', 'qnli']
    sample_sizes = {'train': 50000, 'validation': 5000, 'test': 5000}

    split_mapping = {
        'mnli': {
            'train': ['train'],
            'validation': ['validation_matched', 'validation_mismatched'],
            "test": ["test_matched", "test_mismatched"]

        },
        'default': {
            'train': ['train'],
            'validation': ['validation'],
            'test': ['test']
        }
    }

    local_finder_addition = "/Users/anna/sftp_mount/hpc_disk/"
    out_dir = os.path.join(local_finder_addition, "02-awegmann/TOKENIZER/data/eval-corpora/multi-DIALECT")

    for split in ['train', 'validation', 'test']:
        print(f"Processing split: {split}")

        # Compute the total number of available samples across tasks
        total_available_samples = 0
        task_splits = {}
        for task in tasks:
            # Determine the actual splits for the task
            if task == 'mnli':
                actual_splits = split_mapping['mnli'][split]
            else:
                actual_splits = split_mapping['default'][split]

            # Load the original and perturbed datasets
            original_dataset = load_dataset('nyu-mll/glue', task)

            # perturbed_dataset_path = f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/{task}_multi"
            perturbed_dataset_path = os.path.join(local_finder_addition,
                                                  f"02-awegmann/TOKENIZER/data/eval-corpora/multi-VALUE/{task}_multi")
            # load .csv files and make it a datasets dataset
            perturbed_dataset = load_dataset('csv',
                                             data_files=
                                             {f"{actual_split}":
                                                  os.path.join(perturbed_dataset_path, f"{actual_split}.csv")
                                              for actual_split in actual_splits})

            # Initialize combined splits
            combined_original_split = None
            combined_perturbed_split = None

            for actual_split in actual_splits:
                if actual_split not in original_dataset:
                    print(f"Split {actual_split} not found for task {task}, skipping.")
                    continue

                original_split = original_dataset[actual_split]
                perturbed_split = perturbed_dataset[actual_split]

                # remove entries from original and perturbed dataset at locations where
                #   "dialect_used" is SAE in perturbed dataset
                filtered_original_split = original_dataset[actual_split].filter(
                    lambda x: perturbed_split[x['idx']]['dialect_used'] != 'SAE')
                filtered_perturbed_split = perturbed_dataset[actual_split].filter(lambda x: x['dialect_used'] != 'SAE')
                # Ensure datasets are aligned
                assert len(filtered_original_split) == len(filtered_perturbed_split)
                # print how many samples were removed
                print(f"Removed {(len(original_split) - len(filtered_original_split))/len(original_split)} samples "
                      f"for {task} {actual_split} because of multi-VALUE transformation errors.")

                # Combine splits if necessary
                if combined_original_split is None:
                    combined_original_split = filtered_original_split
                    combined_perturbed_split = filtered_perturbed_split
                else:
                    combined_original_split = concatenate_datasets([combined_original_split, filtered_original_split])
                    combined_perturbed_split = concatenate_datasets(
                        [combined_perturbed_split, filtered_perturbed_split])

            if combined_original_split is None:
                print(f"No valid splits found for task {task} and split {split}, skipping.")
                continue

            task_splits[task] = (combined_original_split, combined_perturbed_split)
            total_available_samples += len(combined_original_split)

        if total_available_samples == 0:
            print(f"No data available for split {split}, skipping.")
            continue

        # flatten all tasks & labels, could use this to make a balanced sample over tasks as well
        total_texts = []
        total_labels = []
        total_tasks = []
        for task, (original_split, perturbed_split) in task_splits.items():
            for example in perturbed_split:
                total_texts.append(get_text(example, task))
                total_labels.append(example['dialect_used'])
                total_tasks.append(task)
            for example in original_split:
                total_texts.append(get_text(example, task))
                total_labels.append('SAE')
                total_tasks.append(task)

        # Collect samples
        #   Calculate the minimum number of samples for each label
        label_counts = {}
        for label in total_labels:
            if label != "SAE":
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        min_samples_per_label = min(label_counts.values())
        print(f"Maximum balanced samples per label: {min_samples_per_label}")
        # for the given split, get the number of samples per label
        samples_per_label = sample_sizes[split] // (len(label_counts) + 1)
        print(f"Actual samples per label: {samples_per_label}")
        assert samples_per_label <= min_samples_per_label

        # Prepare stratified sampling: Sample the same, maximally possible, number of examples for each label
        sampled_texts = []
        sampled_labels = []
        samples_tasks = []
        # shuffle indices
        indices = list(range(len(total_texts)))
        random.seed(42)
        random.shuffle(indices)

        for label in list(label_counts.keys()) + ["SAE"]:
            label_indices = [i for i in indices if total_labels[i] == label]
            label_indices = label_indices[:samples_per_label]
            sampled_texts.extend([total_texts[i] for i in label_indices])
            sampled_labels.extend([total_labels[i] for i in label_indices])
            samples_tasks.extend([total_tasks[i] for i in label_indices])


        # print number of samples per task
        samples_per_task = {}
        for task in tasks:
            samples_per_task[task] = samples_tasks.count(task)
        print(f"Samples per task: {samples_per_task}")
        # print label dist
        label_dist = {}
        for label in list(label_counts.keys()) + ["SAE"]:
            label_dist[label] = sampled_labels.count(label)
        print(f"Label distribution: {label_dist}")

        # Create dataframe
        df = pd.DataFrame({'text': sampled_texts, 'label': sampled_labels})
        # shuffle with resetting index
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to CSV
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)

        print(f"Saved {len(df)} samples for {split}")
        print(f"Saved to {os.path.join(out_dir, f'{split}.csv')}")


if __name__ == '__main__':
    create_datasets()
