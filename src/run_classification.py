#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
import shutil
from datetime import datetime

# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
"""
    copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py
    --> some changes for saving predictions, adapting to datasets etc.
"""
from styletokenizer.utility.env_variables import set_cache

set_cache()

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.45.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a TSV/CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a TSV/CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A tsv, csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A tsv, csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None,
                                     metadata={"help": "A tsv, csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["tsv", "csv", "json"], "`train_file` should be a tsv, csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (tsv, csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # create output dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    if (len(os.listdir(training_args.output_dir)) > 0) and (not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own TSV/CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # test if dataset_name is a path that exists
        if os.path.exists(data_args.dataset_name):
            raw_datasets = load_from_disk(data_args.dataset_name)
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
    else:
        # Loading a dataset from your local files.
        # TSV/CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own TSV/CSV/JSON test file
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (tsv, csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets = check_for_multilabel(raw_datasets, column=data_args.label_column_name)
        elif data_args.train_file.endswith(".tsv"):
            # Loading a dataset from local tsv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                delimiter="\t",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets = check_for_multilabel(raw_datasets, column=data_args.label_column_name)

        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split] = raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        # Make sure we actually have columns to work with
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")

            # If we have exactly 2 columns and the delimiter is '[SEP]',
            # then we let the tokenizer handle them as text + text_pair
            if len(text_column_names) == 2 and data_args.text_column_delimiter == "[SEP]":
                # Safely extract each text column, converting None to ""
                text1 = [str(x) if x is not None else "" for x in examples[text_column_names[0]]]
                text2 = [str(x) if x is not None else "" for x in examples[text_column_names[1]]]

                # Tokenize with text and text_pair -> actual [SEP] token is inserted
                result = tokenizer(
                    text1,
                    text2,
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True,
                )

            else:
                # Fallback: your original join-logic
                # Initialize "sentence" with the first column (convert None -> "")
                examples["sentence"] = [
                    str(x) if x is not None else "" for x in examples[text_column_names[0]]
                ]

                # For each additional column, append using data_args.text_column_delimiter
                for column in text_column_names[1:]:
                    for i in range(len(examples[column])):
                        try:
                            # original code had a comment about using += ...
                            # but currently it's assigning =, so we keep it as-is
                            examples["sentence"][i] += data_args.text_column_delimiter + str(examples[column][i])
                        except TypeError:
                            if examples["sentence"][i] is None:
                                logging.debug(
                                    f"DEBUG: column {text_column_names[0]} is None, "
                                    "replacing with empty string"
                                )
                                examples["sentence"][i] = (
                                        data_args.text_column_delimiter + str(examples[column][i])
                                )
                            elif examples[column][i] is None:
                                logging.debug(
                                    f"DEBUG: column {column} is None, replacing with empty string"
                                )
                                examples["sentence"][i] += data_args.text_column_delimiter + ""
                            else:
                                print(
                                    f"DEBUG: column: {column}, i: {i}, "
                                    f"examples[column]: {examples[column][i]}, "
                                    f'examples["sentence"]: {examples["sentence"][i]}'
                                )

                # Now tokenize the "sentence" field
                result = tokenizer(
                    examples["sentence"],
                    padding=padding,
                    max_length=max_seq_length,
                    truncation=True,
                )

        else:
            # If no text_column_names are provided, we cannot proceed
            raise ValueError("No text_column_names specified in data_args.")

        # Handle labels if they exist
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [
                    (label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]
                ]

        return result

    def prepare_dataset(raw_datasets, dataset_keys, do_flag, max_samples=None, shuffle=False, seed=None, desc=None,
                        preprocess_function=None, num_proc=None, load_from_cache_file=False):
        if not do_flag:
            return None

        # Find the appropriate dataset key
        for key in dataset_keys:
            if key in raw_datasets:
                dataset = raw_datasets[key]
                break
        else:
            raise ValueError(f"Required dataset not found. Available keys: {list(raw_datasets.keys())}")

        # Shuffle the dataset if required
        if shuffle:
            logger.info(f"Shuffling the {desc} dataset")
            dataset = dataset.shuffle(seed=seed)

        # Select a subset of the dataset if max_samples is specified
        if max_samples is not None:
            max_samples = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples))

        # Preprocess only the selected samples
        with training_args.main_process_first(desc="dataset map pre-processing"):
            dataset = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                desc=f"Running tokenizer on {desc} dataset",
            )

        return dataset

    # Prepare the training dataset
    train_dataset = prepare_dataset(
        raw_datasets=raw_datasets,
        dataset_keys=["train"],
        do_flag=training_args.do_train,
        max_samples=data_args.max_train_samples,
        shuffle=data_args.shuffle_train_dataset,
        seed=data_args.shuffle_seed,
        desc="training",
        preprocess_function=preprocess_function,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    # Prepare the evaluation dataset
    eval_dataset = prepare_dataset(
        raw_datasets=raw_datasets,
        dataset_keys=["validation", "validation_matched"],
        do_flag=training_args.do_eval,
        max_samples=data_args.max_eval_samples,
        desc="evaluation",
        preprocess_function=preprocess_function,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    # Prepare the prediction dataset
    predict_dataset = prepare_dataset(
        raw_datasets=raw_datasets,
        dataset_keys=["test", "test_matched"],
        do_flag=training_args.do_predict or data_args.test_file is not None,
        max_samples=data_args.max_predict_samples,
        desc="prediction",
        preprocess_function=preprocess_function,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
            if is_multi_label
            else evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
                logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
            print(f"DEBUG: p.label_ids: {p.label_ids}, sum: {np.sum(p.label_ids, axis=1)}")
            print(f"DEBUG: preds: {preds}, sum: {np.sum(preds, axis=1)}")
        else:
            preds = np.argmax(preds, axis=1)
            print(f"DEBUG: p.label_ids: {p.label_ids}")
            if len(p.label_ids) > 1:
                # check if metric has "average" parameter
                if "average" in metric.compute.__code__.co_varnames:
                    result = metric.compute(predictions=preds, references=p.label_ids, average="macro")
                else:
                    result = metric.compute(predictions=preds, references=p.label_ids)
            else:
                result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Predict again on eval dataset and save all predictions with IDs
        predictions = trainer.predict(eval_dataset)
        eval_dataset = eval_dataset.add_column("predictions",
                                               predictions.predictions.argmax(-1))  # assuming classification
        # Dataset as a TSV file
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_predict_file = os.path.join(training_args.output_dir, f"{current_date}_eval_dataset.tsv")
        eval_dataset.to_csv(output_predict_file, sep='\t', index=False)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        elif is_multi_label:
            # Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
            # You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
            # and set p > 0.5 below (less efficient in this case)
            predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
        else:
            predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    elif is_multi_label:
                        # recover from multi-hot encoding
                        item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                        writer.write(f"{index}\t{item}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # ADDED
    # remove all folders in the model path that have "checkpoint" in their name
    # this is to avoid saving all the checkpoints
    deleted_folders = []
    # List all entries in the folder
    for entry in os.listdir(training_args.output_dir):
        entry_path = os.path.join(training_args.output_dir, entry)

        # Check if it's a directory and if 'checkpoint' is in its name
        if os.path.isdir(entry_path) and "checkpoint" in entry:
            # Delete the directory
            shutil.rmtree(entry_path)
            deleted_folders.append(entry_path)

    # Logging
    if deleted_folders:
        logger.info("Deleted checkpoint folders:")
        for folder in deleted_folders:
            logger.info(f"  - {folder}")
    else:
        logger.info("No checkpoint folders found.")


def check_for_multilabel(raw_datasets, column='label'):
    # if it has a label column, load it as list, IF it is a list in string form (e.g. "[1,2,3]")
    first_element = raw_datasets['train'][0]  # Assuming you are loading 'train'
    # Check if the first element in 'col_with_lists' is a string that looks like a list
    first_value = first_element[column]
    if isinstance(first_value, str) and first_value.strip().startswith('[') and first_value.strip().endswith(
            ']'):

        def convert_to_list(example):
            value = example[column]
            # Try to convert the string to a list only if it looks like a list
            if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
                try:
                    evaluated_value = eval(value)
                    if isinstance(evaluated_value, list):
                        example[column] = evaluated_value
                except:
                    # If eval fails or the string is not convertible to a list, keep it as is
                    pass
            return example

        # Apply the conversion only if the first element looks like a list
        raw_datasets = raw_datasets.map(convert_to_list)
    return raw_datasets


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
