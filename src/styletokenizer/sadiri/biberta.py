import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AutoModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

sys.path.append('../..')

# from src.hiatus_training.losses import InfoNCE_loss_full
# from src.hiatus_training.evaluation import compute_ranking_metrics
from src.custom_training.contrastive_style.trainer_utils import load_model, load_all_dataloaders, alternate_loaders


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        hidden_states = self.dense(sequence_output)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        # The prediction_scores are the logits representing the probability distribution over the vocabulary
        prediction_scores = self.decoder(hidden_states)
        return prediction_scores


class ContrastiveStyleMLM(nn.Module):
    def __init__(self, args, biber_plus_size=192, adversarial_loss_weight=0.1, wandb=None):
        super().__init__()
        self.args = args
        self.style_dimensions = args.style_dimensions
        self.adversarial_loss_weight = adversarial_loss_weight
        self.biber_plus_size = biber_plus_size
        self.device = args.device
        self.loaders = load_all_dataloaders(args)
        self.wandb = wandb

        for name, loader in self.loaders.items():
            print(f"{name} batches: {len(loader)}")

    def init_model(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=self.args.grad_acc > 1)
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.grad_acc, kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device

        if self.args.resume:
            checkpoint_dir = os.path.join(self.args.out_dir, f'{self.args.run_name}', 'last')
            self.model, optimizer, scheduler = load_model(checkpoint_dir)
        else:
            # Initialize the base model and MLM head
            self.model = AutoModel.from_pretrained(self.args.pretrained_model)
            self.mlm_head = MLMHead(self.model.config)
            optimizer = AdamW(list(self.model.parameters()) + list(self.mlm_head.parameters()),
                              lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
            total_steps = sum([len(loader) for _, loader in self.loaders.items() if 'train' in _]) * self.args.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.args.num_warmup_steps, total_steps)

        self.style_ll = nn.Linear(self.style_dimensions, self.biber_plus_size)
        self.not_style_ll = nn.Linear(self.model.config.hidden_size - self.style_dimensions, self.biber_plus_size)
        self.cls_dropout = nn.Dropout(p=0.2)

        (self.model,
         self.mlm_head,
         self.style_ll,
         self.not_style_ll,
         self.optimizer,
         self.loaders['contrastive_train'],
         self.loaders['style_train'],
         self.scheduler) = self.accelerator.prepare(
            self.model,
            self.mlm_head,
            self.style_ll,
            self.not_style_ll,
            optimizer,
            self.loaders['contrastive_train'],
            self.loaders['style_train'],
            scheduler
        )

    # def train_model(self):
    #     self.init_model()
    #     self.model.train()

    #     losses = {"contrastive": 0, "mlm": 0, "style": 0, "adversarial": 0}
    #     best_eval_metric = float('inf')
    #     total_steps = sum([len(loader) for _, loader in self.loaders.items() if 'train' in _])

    #     for epoch in range(self.args.epochs):
    #         for i, (batch, batch_type) in tqdm(
    #                 enumerate(alternate_loaders(self.loaders['contrastive_train'], self.loaders['style_train'])),
    #                 total=total_steps):
    #             with self.accelerator.accumulate(self.model):
    #                 if batch_type == 'contrastive':
    #                     query_batch, candidate_batch = batch
    #                     losses["contrastive"] += self.contrastive_train_step(query_batch, candidate_batch).item()
    #                 else:
    #                     batch, biber_encoding = batch
    #                     total_loss, mlm_loss, style_loss, adversarial_loss = self.mlm_style_step(batch, biber_encoding)
    #                     for name, val in zip(["mlm", "style", "adversarial"], [mlm_loss, style_loss, adversarial_loss]):
    #                         losses[name] += val.item()

    #             if i % self.args.grad_acc == 0 and i > 0:
    #                 self.log_losses(losses)
    #                 losses = {key: 0 for key in losses}

    #             best_eval_metric = self.checkpoint_model(i, best_eval_metric)

    #     self.accelerator.end_training()

    # def contrastive_train_step(self, query_batch, candidate_batch):
    #     z1 = self.model(**query_batch).pooler_output
    #     z2 = self.model(**candidate_batch).pooler_output
    #     loss = InfoNCE_loss_full(z1, z2) / self.args.grad_acc
    #     self.optimizer_step(loss)
    #     return loss

    # def mlm_style_step(self, batch, biber_encoding):
    #     if not isinstance(biber_encoding, torch.Tensor):
    #         biber_encoding = torch.tensor(biber_encoding, device=self.device, dtype=torch.float32)

    #     # Separate input ids and attention masks from labels
    #     inputs = {k: v for k, v in batch.items() if k != 'labels'}
    #     labels = batch['labels']

    #     # Get the base model outputs without MLM head
    #     outputs = self.model(**inputs, output_hidden_states=True)
    #     cls = self._get_cls(outputs)
    #     sequence_output = outputs.hidden_states[-1]

    #     # Apply the MLM head to get prediction scores for MLM loss
    #     prediction_scores = self.mlm_head(sequence_output)

    #     mlm_loss_fct = nn.CrossEntropyLoss()
    #     # When using DistributedDataParallel, you should access the original model's config like this:
    #     vocab_size = self.model.module.config.vocab_size if hasattr(self.model,
    #                                                                 'module') else self.model.config.vocab_size
    #     # Flatten the output to align with labels
    #     mlm_loss = mlm_loss_fct(prediction_scores.view(-1, vocab_size), labels.view(-1))

    #     style_loss, adversarial_loss = self.get_biber_losses(cls, biber_encoding)
    #     total_loss = mlm_loss + style_loss + adversarial_loss

    #     self.optimizer_step(total_loss)
    #     return total_loss, mlm_loss, style_loss, adversarial_loss

    # def checkpoint_model(self, step: int, best_eval_metric: float):
    #     if step % (self.args.saving_step * self.args.grad_acc) == 0 and step > 0:
    #         if self.args.evaluate:
    #             eval_contrastive_loss = self.evaluate_contrastive()
    #             self.evaluate_style()
    #             version = 'best' if eval_contrastive_loss < best_eval_metric else 'last'
    #             self.save_model(step=str(step), version=version)
    #             if version == 'best':
    #                 best_eval_metric = eval_contrastive_loss
    #     return best_eval_metric

    # def log_losses(self, losses: dict):
    #     if self.wandb:
    #         log = {f"Train {key.capitalize()} Loss": val / self.args.grad_acc for key, val in losses.items()}
    #         self.wandb.log(log)
    #         print(log)

    # @staticmethod
    # def _get_cls(outputs):
    #     # Get the last hidden state and take the cls token
    #     return outputs.hidden_states[-1][:, 0, :]

    # def get_biber_losses(self, cls, biber_encoding):
    #     style = cls[:, :self.style_dimensions]
    #     not_style = cls[:, self.style_dimensions:]
    #     style = self.cls_dropout(style)
    #     not_style = self.cls_dropout(not_style)
    #     style_logits = self.style_ll(style)
    #     not_style_logits = self.not_style_ll(not_style)

    #     criterion = nn.CrossEntropyLoss()
    #     style_loss = criterion(style_logits, torch.argmax(biber_encoding, dim=1))
    #     adversarial_loss = criterion(not_style_logits, torch.argmax(biber_encoding, dim=1))

    #     return style_loss, adversarial_loss * self.adversarial_loss_weight

    # def optimizer_step(self, loss):
    #     loss.backward()
    #     self.optimizer.step()
    #     self.scheduler.step()
    #     self.optimizer.zero_grad()

    # def evaluate_contrastive(self):
    #     self.model.eval()
    #     queries, targets = [], []
    #     all_query_authors, all_target_authors = [], []
    #     loader = self.loaders.get('contrastive_dev', None)
    #     if loader:
    #         for query_batch, candidate_batch, query_authors, target_authors in tqdm(loader, total=len(loader),
    #                                                                                 position=0, leave=True):
    #             query = self.model(**query_batch.to(self.device)).pooler_output
    #             target = self.model(**candidate_batch.to(self.device)).pooler_output
    #             queries.append(query.cpu().detach().numpy())
    #             targets.append(target.cpu().detach().numpy())
    #             all_query_authors += query_authors
    #             all_target_authors += target_authors

    #     all_target_authors = np.array(all_target_authors)
    #     all_query_authors = np.array(all_query_authors)
    #     queries = np.concatenate(queries, axis=0)
    #     targets = np.concatenate(targets, axis=0)
    #     results = compute_ranking_metrics(queries, targets, all_query_authors, all_target_authors, 'cosine')
    #     if self.wandb:
    #         eval_log = {f"Eval {key.capitalize()}": val for key, val in results.items()}
    #         wandb.log(eval_log)
    #     self.model.train()
    #     return -1 * results['MRR']

    # def evaluate_style(self, eval_percent=0.05):
    #     self.model.eval()
    #     loader = self.loaders.get('style_dev', None)
    #     if loader:
    #         num_batches = int(len(loader) * eval_percent)
    #         # Add your evaluation code here.
    #         self.model.train()

    # def save_model(self, step, version='last'):
    #     checkpoint_dir = os.path.join(self.args.out_dir, f'{self.args.run_name}', f'{version}')
    #     os.makedirs(checkpoint_dir, exist_ok=True)

    #     # Save the LM
    #     self.accelerator.wait_for_everyone()
    #     unwrapped_model = self.accelerator.unwrap_model(self.model)
    #     unwrapped_model.save_pretrained(checkpoint_dir)

    #     # Save the optimizer and scheduler
    #     torch.save({
    #         'step': step,
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'scheduler_state_dict': self.scheduler.state_dict(),
    #     }, os.path.join(checkpoint_dir, 'optimizer_and_scheduler.pt'))

    #     print(f"Saved {version} model to {checkpoint_dir}")
