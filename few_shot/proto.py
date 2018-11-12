import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss

from few_shot.utils import pairwise_distances


def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      loss_fn: Loss,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      distance: str,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.

    # Arguments
        model:
    """
    if train:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    prototypes = compute_prototypes(support, k_way, n_shot)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, distance)

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss.item(), y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    return support.reshape(k, n, -1).mean(dim=1)