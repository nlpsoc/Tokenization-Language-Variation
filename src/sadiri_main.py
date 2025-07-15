"""
    contrastive learning approach to authorship verification; taken from UMich project
"""
from styletokenizer.utility.env_variables import set_cache
set_cache()

import os
import re
import json
import wandb
import argparse
from datasets import load_from_disk
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer, set_seed
from styletokenizer.sadiri.data_utils import TextDataCollator, AATrainData
from styletokenizer.sadiri.cluster_batches import ClusterData
from styletokenizer.sadiri.load_model import load_model
from styletokenizer.sadiri.trainer import Trainer
from styletokenizer.sadiri.losses import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    torch.autograd.set_detect_anomaly(True)

    # torch.autograd.set_detect_anomaly(True)

    ############# SET WANDB ############
    if args.wandb:
        wandb.init(project=args.project_name,
                   settings=wandb.Settings(code_dir=args.code_dir),
                   config={"epochs": args.epochs,
                           "batch_size": args.batch_size,
                           "eval_batch_size": args.eval_batch_size,
                           "max_length": args.max_length,
                           "saving_step": args.saving_step,
                           "loss": args.loss,
                           "grad_norm": args.grad_norm,
                           "learning_rate": args.learning_rate,
                           "pretrained_model": args.pretrained_model,
                           "gradient_accumulation": args.grad_acc})
        wandb.run.name = args.run_name

    ############# SET SEED ############
    set_seed(args.seed)

    ############# LOAD TOKENIZER ############
    print("intialize the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ############# LOAD MODEL ############
    model = load_model(args, tokenizer)
    model = model.to(device)

    ############# LOAD DATASET ############
    if args.train:
        print("loading training data...")
        train_dataset = load_from_disk(args.train_data)

        if args.num_training_samples > 1:
            train_dataset = train_dataset.select(
                [i for i in range(args.num_training_samples)]).shuffle(seed=args.seed)

        train_collator = TextDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            mask=args.mask,
            path=args.corpus,
            top=args.top)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_collator)

        cluster = ClusterData(
            batch_size=args.batch_size,
            batch_count=len(train_dataloader),
            shuffle=True,
            seed=args.seed
        )
        print("training data loaded")

        print('loading dev data...')
        dev_dataset = load_from_disk(args.dev_data).shuffle(seed=args.seed)

        if args.num_eval_samples > 1:
            dev_dataset = dev_dataset.select(
                [i for i in range(args.num_eval_samples)])

        dev_collator = TextDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            evaluate=True,
            mask=0,
            path=args.corpus,
            top=args.top)

        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=dev_collator)
        print("dev data loaded")

        print("There are %s training batches." % len(train_dataloader))
        print("There are %s evaluation batches" % len(dev_dataloader))
        print("initialize the model with pretrained weights...")

    if args.evaluate:

        print('loading test data...')
        test_dataset = load_from_disk(args.test_data)['train'].shuffle(seed=args.seed)

        if args.num_eval_samples > 1:
            test_dataset = test_dataset.select([i for i in range(args.num_eval_samples)])

        test_collator = TextDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            evaluate=True,
            mask=0,
            path=args.corpus,
            top=args.top)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=test_collator)
        print("test data loaded")

        print("There are %s test batches." % len(test_dataloader))

    os.makedirs(args.out_dir, exist_ok=True)
    trainer = Trainer(args)

    ############# START TRAINING ############
    print('\t-----------------------------------------------------------------------')
    print(f'\t ============= Start Model Training =============')
    print('\t-----------------------------------------------------------------------')
    if args.train:
        train_data = AATrainData(
            dataloader=train_dataset,
            clustering=cluster,
            batch_size=args.batch_size
        )
        print("#######################")
        print(args.top)
        trainer.train(model, train_data, dev_dataloader, train_collator)

    ############# START EVALUATION ############
    if args.evaluate:
        results = trainer.evaluate(model, test_dataloader)
        print('\t-----------------------------------------------------------------------')
        print(f'\t ============= Evaluation Results =============')
        print('\t-----------------------------------------------------------------------')
        print(results)
        print(type(results))
        converted_data = {k: float(v) for k, v in results.items()}
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
            json.dump(converted_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data', default='/shared/3/projects/hiatus/pretraining/data/train', type=str)
    parser.add_argument(
        '--dev_data', default='/shared/3/projects/hiatus/pretraining/data/dev', type=str)
    parser.add_argument(
        '--test_data', default='/shared/3/projects/hiatus/pretraining/data/test', type=str)
    parser.add_argument('--out_dir', type=str,
                        default='/shared/3/projects/hiatus/pretraining/models/models')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--pretrained_model', default='roberta-base', type=str)
    parser.add_argument('--tokenizer', default=None, type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--grad_norm', default=1.0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--grad_acc', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step', default=200, type=int)  # note that this is currently tied to
    # the eval steps, i.e., if saving_steps is set to 0, then the model is never evaluated on the dev set,
    # this only ever saves the best model
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--max_length', default=350, type=int)
    parser.add_argument('--loss', default='contrastive_full', type=str)
    parser.add_argument('--gradient_checkpointing', default=False, type=bool)
    parser.add_argument('--num_warmup_steps', default=1000, type=int)
    parser.add_argument('--num_training_samples', default=-1, type=int)
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--multivector', action='store_true')  # never used for this project
    parser.add_argument('--sparse', action='store_true')

    parser.add_argument('--eval_name', type=str, default='reddit')
    parser.add_argument('--metric', default='cosine', type=str)
    parser.add_argument('--regularization', default='l1', type=str)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--cluster', default=False, action='store_true')

    # the following arguments are only relevant if you hope to log results in wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--entity', default="sadiri-michigan", type=str)
    parser.add_argument('--project_name', default='Hiatus-TA1', type=str)
    parser.add_argument('--run_name', default='V1', type=str)
    parser.add_argument('--code_dir', default='src', type=str)

    # parameters for masking
    parser.add_argument('--mask', default=0.8, type=float)
    parser.add_argument('--top', default=200, type=int)
    parser.add_argument('--corpus', type=str,
                        default="/shared/3/projects/hiatus/aggregated_trainset_v2/content_masking_research/down_1_shuffle/aggregate")

    # parameters for seed
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main(args)
