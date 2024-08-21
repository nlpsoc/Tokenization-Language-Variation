"""
    functions copied from SADIRI project
"""
import torch
from pytorch_metric_learning import losses


class Loss:

    def __call__(self, representations: torch.Tensor, class_ids: torch.Tensor):
        raise NotImplementedError


class LossContrastive(Loss):

    def __init__(self, temperature: float):
        self.loss_fn = losses.SupConLoss(temperature=temperature)

    def __call__(self, representations: torch.Tensor, class_ids: torch.Tensor):
        return self.loss_fn(representations, class_ids)


def SupConLoss_positive(z1, z2, temperature=0.05):
    """
        expects z1 and z2 to be parallel with positive classes, i.e.,
            row 1 of z1 nad row 1 of z2 are a positive pair, everything else is a negative
        copied from:
            https://github.com/davidjurgens/sadiri/blob/2f9bee96d41d344f28574a1123c4e02b2c1efc30/src/custom_training/content_word_masking/losses.py
    """
    loss_fn = LossContrastive(temperature=temperature)
    concatenated_tensor = torch.cat([z1, z2], dim=0)

    # Create labels
    labels = torch.arange(z1.size(0)).long().to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    return loss_fn(concatenated_tensor, labels)
