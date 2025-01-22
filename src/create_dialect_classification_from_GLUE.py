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

        # Determine sample size (number of sentence pairs)
        max_samples = sample_sizes.get(split, total_available_samples)
        sample_size = min(max_samples, total_available_samples)

        print(f"Total available sentence pairs for {split}: {total_available_samples}")
        print(f"Sample size (number of sentence pairs) for {split}: {sample_size}")

        # Calculate number of samples per task proportionally
        samples_per_task = {}
        for task, (original_split, perturbed_split) in task_splits.items():
            task_length = len(original_split)
            proportion = task_length / total_available_samples
            samples_per_task[task] = int(proportion * sample_size)

        # Adjust for rounding errors
        total_assigned_samples = sum(samples_per_task.values())
        difference = sample_size - total_assigned_samples
        if difference > 0:
            # Assign remaining samples randomly to tasks
            tasks_list = list(samples_per_task.keys())
            for _ in range(difference):
                task = random.choice(tasks_list)
                samples_per_task[task] += 1

        print(f"Samples per task for {split}: {samples_per_task}")

        # Collect samples
        #   Calculate the minimum number of samples for each label
        label_counts = {}
        for task, (original_split, perturbed_split) in task_splits.items():
            for example in perturbed_split:
                label = example['dialect_used']
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        min_samples_per_label = min(label_counts.values())

        # Sample the same, maximally possible, number of examples for each label
        total_texts = []
        total_labels = []
        for task, (original_split, perturbed_split) in task_splits.items():
            label_samples = {label: [] for label in label_counts.keys()}

            for example in perturbed_split:
                label = example['dialect_used']
                if len(label_samples[label]) < min_samples_per_label:
                    label_samples[label].append(example)

            for label, samples in label_samples.items():
                indices = list(range(len(samples)))
                random.seed(42)
                random.shuffle(indices)
                indices = indices[:min_samples_per_label]

                sampled_texts = [get_text(samples[i], task) for i in indices]
                total_texts.extend(sampled_texts)
                total_labels.extend([label] * min_samples_per_label)

            # Add 'SAE' samples from the original dataset
            original_indices = list(range(len(original_split)))
            random.seed(43)
            random.shuffle(original_indices)
            original_indices = original_indices[:min_samples_per_label]

            original_texts = [get_text(original_split[i], task) for i in original_indices]
            total_texts.extend(original_texts)
            total_labels.extend(['SAE'] * min_samples_per_label)

        # Ensure the total number of samples equals the previously calculated samples_per_task
        total_samples = sum(samples_per_task.values())
        if len(total_texts) > total_samples:
            indices = list(range(len(total_texts)))
            random.seed(44)
            random.shuffle(indices)
            indices = indices[:total_samples]
            total_texts = [total_texts[i] for i in indices]
            total_labels = [total_labels[i] for i in indices]
        elif len(total_texts) < total_samples:
            raise ValueError("Not enough samples collected.")

        # Combine and shuffle
        combined = list(zip(total_texts, total_labels))
        random.seed(45)
        random.shuffle(combined)
        total_texts[:], total_labels[:] = zip(*combined)

        # Create dataframe
        df = pd.DataFrame({'text': total_texts, 'label': total_labels})

        # Save to CSV
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)

        print(f"Saved {len(df)} samples for {split}")
        print(f"Saved to {os.path.join(out_dir, f'combined_{split}.csv')}")


if __name__ == '__main__':
    create_datasets()
