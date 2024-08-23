import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses


def splade_max_pool(z, attention_mask=None):
    if attention_mask is not None:
        z[~attention_mask.bool().unsqueeze(2).repeat(1, 1, z.size(2))] = -100
    return torch.log(1 + F.relu(z)).max(dim=1).values[:, 4:-1]


def cosine_similarity(z1, z2, temperture=0.05):
    '''
    Cosine similarity with temperture
    '''
    cos = F.cosine_similarity(z1, z2, dim=-1)
    return cos / temperture


def l1_loss(z):
    z = F.normalize(z, dim=-1)
    return torch.mean(torch.abs(z), dim=-1).mean()

def SupConLoss(z1, z2, temperture=0.05):
    features = torch.cat([z1, z2], dim=0)  # Shape: [2 * batch_size, feature_dim]
    
    loss_fn = losses.SupConLoss(temperature=temperture)
    
    labels = torch.arange(z1.size(0)).long().to(z1.device)
    labels = torch.cat([labels, labels], dim=0)
    loss = loss_fn(features, labels)
    return loss


def InfoNCE_loss(z1, z2, temperture=0.05):
    '''
    InfoNCE loss function
    '''
    loss_fn = nn.CrossEntropyLoss()
    sim = cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), temperture)
    labels = torch.arange(sim.size(0)).long().to(z1.device)
    loss = loss_fn(sim, labels)
    return loss


def Multivector_contrastive_loss(z1, z2, z1_mask, z2_mask, temperture=0.05):
    '''
    InfoNCE loss for multivector retrieval
    '''
    # normalize hidden representations
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    loss_fn = nn.CrossEntropyLoss()
    # compuate pairwise token similarities for all pairs
    out = torch.einsum("bik,cjk->bcij", z1, z2)
    # set padding tokens to 0 
    out[~z1_mask.unsqueeze(1).unsqueeze(3).repeat(1, out.size(1), 1, out.size(3)).bool()] = 0
    out[~z2_mask.unsqueeze(0).unsqueeze(2).repeat(out.size(0), 1, out.size(2), 1).bool()] = 0
    # maxsim operations
    sim1 = out.max(-1).values.sum(-1)
    sim2 = out.transpose(1, 0).transpose(3, 2).max(-1).values.sum(-1)
    # compute loss
    labels = torch.arange(sim1.size(0)).long().to(z1.device)
    loss1 = loss_fn(sim1, labels)
    loss2 = loss_fn(sim2, labels)

    return 0.5 * loss1 + 0.5 * loss2


def InfoNCE_loss_full(z1, z2, temperature=0.05, metric='cosine'):
    '''
    InfoNCE loss function
    '''
    loss_fn = nn.CrossEntropyLoss()
    if metric == 'cosine':
        sim = cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), temperature)
    elif metric == 'dot_product':
        sim = torch.matmul(z1, z2.t())
        print(sim)
    batch_size = sim.size(0)
    I = torch.eye(batch_size).bool()  # mask out diagonal entries

    # add more in-batch negative samples
    sim = torch.concat([sim, sim.t()[~I].reshape(batch_size, batch_size - 1)], dim=1)
    if metric == 'cosine':
        sim_a = cosine_similarity(z1.unsqueeze(1), z1.unsqueeze(0), temperature)
        sim_b = cosine_similarity(z2.unsqueeze(1), z2.unsqueeze(0), temperature)
    elif metric == 'dot_product':
        sim_a = torch.matmul(z1, z1.t())
        sim_b = torch.matmul(z2, z2.t())

    sim = torch.concat([sim, sim_a[~I].reshape(batch_size, batch_size - 1)], dim=1)
    sim = torch.concat([sim, sim_b[~I].reshape(batch_size, batch_size - 1)], dim=1)

    labels = torch.arange(sim.size(0)).long().to(z1.device)
    loss = loss_fn(sim, labels)
    return loss



def max_margin_loss(z1, z2, tau_low=0.2, tau_high=0.8):
    """
    Compute the max margin loss.

    Negative instances are sampled from in-batch samples.

    Reference:

    B. Boenninghoff, S. Hessler, D. Kolossa and R. M. Nickel, "Explainable Authorship
    Verification in Social Media via Attention-based Similarity Learning," 2019 IEEE
    International Conference on Big Data (Big Data), Los Angeles, CA, USA, 2019, pp. 36-45,
    doi: 10.1109/BigData47090.2019.9005650.

    :param z1: Tensor of first documents in the batch.
    :param z2: Tensor of second documents (same author as `z1`) in the batch.
    :param tau_low: distance ceiling for docs with the same author
    :param tau_high: distance floor for docs with differing authors
    :return: Combined loss.
    """

    scores_pos = 1 - F.cosine_similarity(z1, z2, dim=-1)
    # randomly select in-batch samples as negative instances
    # Roll z2 elements down one place so that neg_samples no longer matches z1.
    neg_samples = torch.roll(z2, 1, 0)
    scores_neg = 1 - F.cosine_similarity(z1, neg_samples, dim=-1)

    positive = torch.max(scores_pos - tau_low, torch.tensor(0.0).to(z1.device)) ** 2  # Equation 9
    negative = torch.max(tau_high - scores_neg, torch.tensor(0.0).to(z1.device)) ** 2  # Equation 10

    return torch.mean(positive) + torch.mean(negative)  # Equation 11



class SigmoidLoss(nn.Module):
    '''
    Sigmoid loss function
    '''
    def __init__(self, init_t_prime=1.0, init_b=-5.0):
        super(SigmoidLoss, self).__init__()
        self.t_prime = nn.Parameter(torch.tensor(init_t_prime))
        self.b = nn.Parameter(torch.tensor(init_b))

        self.loss_fn = torch.nn.LogSigmoid()
        print('Initialised k-hard Sigmoid Loss')

    def forward(self, z1, z2, mini_batch_size = 8, temperature=0.05, metric='cosine'):
        batch_size = z1.size(0)
        t = torch.exp(self.t_prime)
        z1_norm = torch.nn.functional.normalize(z1, p=2, dim=1)
        z2_norm = torch.nn.functional.normalize(z2, p=2, dim=1)
        logits = torch.matmul(z1_norm, z2_norm.t()) * t + self.b

        base = torch.zeros(batch_size, batch_size)
        for i in range(batch_size):
            if i % mini_batch_size == 0:
                for m in range(i, i + mini_batch_size):
                    for n in range(i, i + mini_batch_size):
                        base[m][n] = 1
        labels = (2 * base - torch.ones(batch_size)).long().to(z1.device)

        ## old labels for one-pair
        # labels = (2 * torch.eye(batch_size) - torch.ones(batch_size)).long().to(z1.device)  # -1 with diagonal 1
        
        loss = -torch.sum(self.loss_fn(labels * logits)) / batch_size
        return loss





if __name__ == "__main__":
    z1 = torch.rand([64, 768])
    z2 = torch.rand([64, 768])

    loss_function = SigmoidLoss()
    loss_function(z1, z2, temperature=0.05, metric='cosine')


    # InfoNCE_loss_full(z1, z2, temperature=0.05, metric='cosine')