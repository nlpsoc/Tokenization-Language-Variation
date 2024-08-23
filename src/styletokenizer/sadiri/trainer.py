import os
import re
import torch
import wandb
import numpy as np
from styletokenizer.sadiri.losses import *
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from opt_einsum import contract
from accelerate import Accelerator
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, T5Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import logging

from accelerate import DistributedDataParallelKwargs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    """Trains and evaluates the models.
    """

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        # AATrainData

    def train(self, encoder, train_data, dev_loader, train_collator):

        ########## Set Loss Functions ##########
        if self.args.multivector:
            self.args.loss = 'multivector_contrastive'
        if self.args.loss == 'contrastive':
            loss_fn = InfoNCE_loss
        elif self.args.loss == 'contrastive_full':
            loss_fn = InfoNCE_loss_full
        elif self.args.loss == 'max_margin':
            loss_fn = max_margin_loss
        elif self.args.loss == 'multivector_contrastive':
            loss_fn = Multivector_contrastive_loss
        elif self.args.loss == 'k_hard_sigmoid_loss':
            loss_fn = SigmoidLoss()
        elif self.args.loss == 'cosine_similarity':
            loss_fn = cosine_similarity
        elif self.args.loss == 'SupConLoss':
            loss_fn = SupConLoss
        logging.info("Using the %s loss" % self.args.loss)

        # encoder = DDP(encoder, device_ids=[accelerator.local_process_index], find_unused_parameters=True)

        encoder.to(device)

        ########## Set Training Params ##########
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.args.grad_acc)
        optimizer = AdamW(encoder.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.num_warmup_steps,
                                                    num_training_steps=len(train_data) * self.args.epochs, num_cycles=1)

        # encoder = DDP(encoder, device_ids=[accelerator.local_process_index], find_unused_parameters=True)

        # comment temporarily because current no distributed 
        ####################################################
        # encoder, optimizer,  train_loader, scheduler = accelerator.prepare(
        #     encoder, optimizer, train_dataloader, scheduler)

        ########## Set WANDB to monitor ##########
        if self.args.wandb:
            wandb.watch(encoder)

        best_perf = 0
        running_loss = 0
        running_decoder_loss = 0

        for epoch in range(self.args.epochs):
            if (self.args.cluster):
                if (epoch == 0):
                    logging.info("generating representation without model update...")
                    self.run_train_without_model_update(encoder, train_data, train_collator)
                train_data.on_epoch()

            train_dataloader = DataLoader(
                train_data.dataloader,
                batch_size=train_data.batch_size,
                shuffle=False,
                collate_fn=train_collator)

            for i, (batchA, batchB) in tqdm(enumerate(train_dataloader)):
                with accelerator.accumulate(encoder):
                    encoder.train()
                    if re.search(r'RWKV', self.args.pretrained_model):
                        del batchA['attention_mask']
                        del batchB['attention_mask']

                    if self.args.multivector:
                        z1 = encoder(**batchA.to(device)).last_hidden_state
                        z2 = encoder(**batchB.to(device)).last_hidden_state
                        loss = loss_fn(
                            z1, z2, batchA['attention_mask'], batchB['attention_mask']) / self.args.grad_acc

                    elif self.args.sparse:
                        z1 = encoder(**batchA.to(device)).logits
                        z2 = encoder(**batchB.to(device)).logits
                        z1 = splade_max_pool(z1, batchA['attention_mask'])
                        z2 = splade_max_pool(z2, batchB['attention_mask'])
                        loss = loss_fn(
                            z1, z2, metric=self.args.metric) / self.args.grad_acc
                        if self.args.regularization == 'l1':
                            loss = loss + 0.1 * \
                                   (0.5 * l1_loss(z1) + 0.5 * l1_loss(z2))

                    else:
                        z1 = accelerator.unwrap_model(encoder)(**batchA.to(device)).pooler_output
                        z2 = accelerator.unwrap_model(encoder)(**batchB.to(device)).pooler_output

                        loss = loss_fn(z1, z2) / self.args.grad_acc

                    model_output = torch.cat([z1, z2])
                    train_data.clustering.add(model_output)
                    accelerator.backward(loss)
                    running_loss += loss.item()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if i % self.args.grad_acc == 0:
                    if self.args.wandb:
                        wandb.log({"loss": running_loss / self.args.grad_acc})
                        if self.args.decoder:
                            wandb.log(
                                {"decoder_loss": running_decoder_loss / self.args.grad_acc})
                            running_decoder_loss = 0

                    running_loss = 0

                if i != 0 and i % (self.args.saving_step * self.args.grad_acc) == 0:
                    ############# CREATE DIRS ############
                    if not os.path.exists(self.args.out_dir):
                        os.mkdir(self.args.out_dir)
                    # if not os.path.exists(self.args.out_dir + "/last_model"):
                    #     os.mkdir(self.args.out_dir + "/last_model")
                    if not os.path.exists(self.args.out_dir + "/best_model"):
                        os.mkdir(self.args.out_dir + "/best_model")

                    model = accelerator.unwrap_model(encoder)

                    ###### If validated, save the best model ######
                    if self.args.validate:
                        if not self.args.multivector:
                            results = self.evaluate(encoder, dev_loader)
                        else:
                            results = self.evaluate_multivector(encoder, dev_loader)
                        logging.info("====== Validating results:\n", results)

                        if best_perf < results['MRR']:
                            logging.info("===== saving model =====")
                            if hasattr(model, 'save_pretrained'):
                                model.save_pretrained(self.args.out_dir + "/best_model")
                            else:
                                torch.save(model.state_dict(), self.args.out_dir + "/best_model/pytorch_model.pth")
                            best_perf = results['MRR']

        if self.args.validate:
            if not self.args.multivector:
                results = self.evaluate(encoder, dev_loader)
            else:
                results = self.evaluate_multivector(encoder, dev_loader)
            logging.info("====== Validating results:\n", results)
            print("====== Validating results:\n", results)

    def run_train_without_model_update(self, encoder, train_data, train_collator):
        encoder.to(device)
        train_dataloader = DataLoader(
            train_data.dataloader,
            batch_size=train_data.batch_size,
            shuffle=False,
            collate_fn=train_collator)

        for i, (batchA, batchB) in tqdm(enumerate(train_dataloader)):
            with torch.no_grad():
                z1 = encoder(**batchA.to(device)).pooler_output.detach()
                z2 = encoder(**batchB.to(device)).pooler_output.detach()

                model_output = torch.cat([z1, z2])
                train_data.clustering.add(model_output)

    def evaluate(self, encoder, dev_loader):

        encoder.eval()
        print('Evaluating...')
        queries = []
        targets = []
        all_query_authors = []
        all_target_authors = []

        with torch.autocast(device_type="cuda", enabled=True):
            # with torch.no_grad():
            for batchA, batchB, query_authors, target_authors in tqdm(dev_loader):
                with torch.no_grad():
                    if self.args.sparse:
                        query = encoder(**batchA.to(device)).logits
                        target = encoder(**batchB.to(device)).logits
                        query = splade_max_pool(
                            query, batchA['attention_mask'])
                        target = splade_max_pool(
                            target, batchB['attention_mask'])
                    else:
                        query = encoder(**batchA.to(device)).pooler_output
                        target = encoder(**batchB.to(device)).pooler_output

                    queries.append(query.cpu().detach().numpy())
                    targets.append(target.cpu().detach().numpy())
                    all_query_authors += query_authors
                    all_target_authors += target_authors

        all_target_authors = np.array(all_target_authors)
        all_query_authors = np.array(all_query_authors)
        queries = np.concatenate(queries, axis=0)
        targets = np.concatenate(targets, axis=0)

        results = self._compute_ranking_metrics(
            queries, targets, all_query_authors, all_target_authors, self.args.metric)

        if self.args.wandb:
            wandb.log({"Eval MRR": results['MRR']})
            wandb.log({"Eval R@8": results['R@8']})
            wandb.log({"Eval R@50": results['R@50']})
            wandb.log({"Eval R@100": results['R@100']})

        return results

    def evaluate_multivector(self, model, dataloader):
        model.eval()
        model.to(device)
        print('Evaluating...')
        max_len = self.args.max_length
        queries = []
        targets = []
        query_lens = []
        target_mask = []
        all_query_authors = []
        all_target_authors = []

        with torch.autocast(device_type="cuda", enabled=True):
            for batchA, batchB, query_authors, target_authors in tqdm(dataloader):
                with torch.no_grad():
                    query = model(**batchA.to(device)).last_hidden_state
                    target = model(**batchB.to(device)).last_hidden_state
                    query_padded = np.zeros(
                        (query.size(0), max_len, query.size(-1)))
                    target_padded = np.zeros(
                        (target.size(0), max_len, target.size(-1)))
                    query_padded[:, :query.size(1), :] = F.normalize(
                        query, dim=-1).cpu().detach().numpy()
                    target_padded[:, :target.size(1), :] = F.normalize(
                        target, dim=-1).cpu().detach().numpy()
                    query_lens.append(
                        torch.sum(batchA['attention_mask'], dim=-1).cpu().detach().numpy())
                    mask_padded = torch.zeros((target.size(0), max_len))
                    mask_padded[:, :target.size(1)] = batchB['attention_mask']
                    target_mask.append(mask_padded)
                    queries.append(query_padded)
                    targets.append(target_padded)
                    all_query_authors += query_authors
                    all_target_authors += target_authors

        all_target_authors = np.array(all_target_authors)
        all_query_authors = np.array(all_query_authors)
        queries = np.concatenate(queries, axis=0)
        targets = np.concatenate(targets, axis=0)
        query_lens = np.concatenate(query_lens)
        target_mask = np.concatenate(target_mask, axis=0)
        results = self._compute_ranking_metrics_multivector(
            queries, targets, query_lens, target_mask, all_query_authors, all_target_authors)

        if self.args.wandb:
            wandb.log({"Eval MRR": results['MRR']})
            wandb.log({"Eval R@8": results['R@8']})
            wandb.log({"Eval R@50": results['R@50']})
            wandb.log({"Eval R@100": results['R@100']})

        return results

    def _compute_ranking_metrics(
            self,
            queries,
            targets,
            query_authors,
            target_authors,
            metric='cosine'):

        num_queries = len(query_authors)
        print("Computing ranking metrics for {} queries".format(num_queries))
        ranks = np.zeros((num_queries), dtype=np.float32)
        reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)
        distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=6)
        errors = []
        for i in range(num_queries):
            try:
                dist = distances[i]
                sorted_indices = np.argsort(dist)
                sorted_target_authors = target_authors[sorted_indices]
                ranks[i] = np.where(sorted_target_authors == query_authors[i])[0].item()
                reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
            except:
                errors.append(i)

        ranks[errors] = -100
        reciprocal_ranks[errors] = -100
        ranks = ranks[ranks != -100]
        reciprocal_ranks = reciprocal_ranks[reciprocal_ranks != -100]

        return_dict = {
            'MRR': np.mean(reciprocal_ranks),
            'R@8': np.sum(np.less_equal(ranks + 1, 8)) / np.float32(num_queries),
            'R@50': np.sum(np.less_equal(ranks + 1, 50)) / np.float32(num_queries),
            'R@100': np.sum(np.less_equal(ranks + 1, 100)) / np.float32(num_queries),
            'R@1000': np.sum(np.less_equal(ranks + 1, 1000)) / np.float32(num_queries)
        }
        return return_dict

    def _compute_ranking_metrics_multivector(
            self,
            queries,
            targets,
            query_lens,
            target_mask,
            query_authors,
            target_authors,
            metric='cosine',
    ):
        num_queries = len(query_authors)
        ranks = np.zeros((num_queries), dtype=np.float32)
        reciprocal_ranks = np.zeros((num_queries), dtype=np.float32)

        target_mask = torch.tensor(target_mask).bool()
        distances = torch.zeros(len(query_authors), len(target_authors))
        for i in tqdm(range(num_queries)):
            query = queries[i]
            dist = contract("ik,cjk->cij", torch.tensor(queries[i, :query_lens[i]]).to('cuda'),
                            torch.tensor(targets).to('cuda'))
            dist[~target_mask.unsqueeze(1).repeat(1, query_lens[i], 1)] = -100
            distances[i, :] = dist.max(-1).values.sum(-1).cpu()

        distances = distances.detach().numpy()
        print(distances)
        errors = []
        for i in range(num_queries):
            try:
                dist = distances[i]
                sorted_indices = np.argsort(dist)[::-1]
                sorted_target_authors = target_authors[sorted_indices]
                ranks[i] = np.where(sorted_target_authors ==
                                    query_authors[i])[0].item()
                reciprocal_ranks[i] = 1.0 / float(ranks[i] + 1)
            except:
                errors.append(i)
        print("%s error samples" % len(errors))
        ranks[errors] = -100
        reciprocal_ranks[errors] = -100
        ranks = ranks[ranks != -100]
        reciprocal_ranks = reciprocal_ranks[reciprocal_ranks != -100]

        return_dict = {
            'MRR': np.mean(reciprocal_ranks),
            'R@8': np.sum(np.less_equal(ranks + 1, 8)) / np.float32(num_queries),
            'R@50': np.sum(np.less_equal(ranks + 1, 50)) / np.float32(num_queries),
            'R@100': np.sum(np.less_equal(ranks + 1, 100)) / np.float32(num_queries)
        }

        return return_dict
