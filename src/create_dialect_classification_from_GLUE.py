"""
    Generated with ChatGPT o1-preview on 2024-11-07
"""
import os
import random
import pandas as pd
from datasets import load_dataset, load_from_disk

def get_text(example, task):
    if task == 'sst2':
        return example['sentence']
    elif task == 'qqp':
        return example['question1'] + example['question2']
    elif task == 'mnli':
        return example['premise'] + example['hypothesis']
    elif task == 'qnli':
        return example['question'] + example['sentence']
    else:
        return ''

def create_datasets():
    tasks = ['sst2', 'qqp', 'mnli', 'qnli']
    sample_sizes = {'train': 25000, 'validation': 2500, 'test': 2500}

    for split in ['train', 'validation', 'test']:
        print(f"Processing split: {split}")

        total_original_texts = []
        total_perturbed_texts = []

        # Compute the total number of available samples across tasks
        total_available_samples = 0
        task_splits = {}
        for task in tasks:
            # Load the original and perturbed datasets
            original_dataset = load_dataset('nyu-mll/glue', task)
            perturbed_dataset_path = f"/hpc/uu_cs_nlpsoc/02-awegmann/TOKENIZER/data/eval-corpora/value/{task}"
            perturbed_dataset = load_from_disk(perturbed_dataset_path)

            if split not in original_dataset:
                print(f"Split {split} not found for task {task}, skipping.")
                continue

            original_split = original_dataset[split]
            perturbed_split = perturbed_dataset[split]

            # Ensure datasets are aligned
            min_length = min(len(original_split), len(perturbed_split))
            original_split = original_split.select(range(min_length))
            perturbed_split = perturbed_split.select(range(min_length))

            task_splits[task] = (original_split, perturbed_split)
            total_available_samples += min_length

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
        for task, (original_split, perturbed_split) in task_splits.items():
            num_samples = samples_per_task[task]

            indices = list(range(len(original_split)))
            random.seed(42)
            random.shuffle(indices)
            indices = indices[:num_samples]

            original_samples = original_split.select(indices)
            perturbed_samples = perturbed_split.select(indices)

            original_texts = [get_text(x, task) for x in original_samples]
            perturbed_texts = [get_text(x, task) for x in perturbed_samples]

            total_original_texts.extend(original_texts)
            total_perturbed_texts.extend(perturbed_texts)

        # Labels
        texts = total_original_texts + total_perturbed_texts
        labels = [0] * len(total_original_texts) + [1] * len(total_perturbed_texts)

        # Combine and shuffle
        combined = list(zip(texts, labels))
        random.seed(42)
        random.shuffle(combined)
        texts[:], labels[:] = zip(*combined)

        # Create dataframe
        df = pd.DataFrame({'text': texts, 'label': labels})

        # Save to CSV
        output_dir = f"./combined_{split}"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"combined_{split}.csv"), index=False)

        print(f"Saved {len(df)} samples for {split}")

if __name__ == '__main__':
    create_datasets()