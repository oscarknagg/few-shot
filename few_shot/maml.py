import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss
from typing import Dict, List

from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Loss,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: torch.device):
    """
    Perform a gradient step on a meta-learner.

    # Arguments
        model: Base model of the meta-learner. We
        optimiser:
        loss_fn:
        x:
        y:
        n_shot:
        k_way:
        q_queries:
        order:
        inner_train_steps:
        inner_lr:
        train:
        device:
    """
    channels, height, width = x.shape[2:]

    task_gradients = []
    task_losses = []
    task_predictions = []
    for meta_batch in x:
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values())

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True if order == 2 else False)
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)

    if train:
        if order == 1:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros(k_way, channels, height, width).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        elif order == 2:
            model.train()
            optimiser.zero_grad()
            meta_batch_loss = torch.stack(task_losses).mean()
            meta_batch_loss.backward()
            optimiser.step()
        else:
            raise ValueError('Order must be either 1 or 2.')

    return torch.stack(task_losses).mean().item(), torch.cat(task_predictions)


# def apply_meta_update(order: int, meta_model: Module, loss_fn: Loss, optimiser: Optimizer,
#                       k_way: int, channels: int, height: int, width: int,
#                       task_gradients: List[Dict[str, torch.Tensor]], task_losses: List[float],
#                       device: torch.device):
#     """"""
#     if order == 1:
#         sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
#                               for k in task_gradients[0].keys()}
#         hooks = []
#         for name, param in meta_model.named_parameters():
#             hooks.append(
#                 param.register_hook(replace_grad(sum_task_gradients, name))
#             )
#
#         meta_model.train()
#         optimiser.zero_grad()
#         # Dummy pass in order to create `loss` variable
#         # Replace dummy gradients with mean task gradients using hooks
#         logits = meta_model(torch.zeros(k_way, channels, height, width).to(device, dtype=torch.double))
#         loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
#         loss.backward()
#         optimiser.step()
#
#         for h in hooks:
#             h.remove()
#
#     elif order == 2:
#         meta_model.train()
#         optimiser.zero_grad()
#         meta_batch_loss = torch.stack(task_losses).mean()
#         meta_batch_loss.backward()
#         optimiser.step()
#     else:
#         raise ValueError('Order must be either 1 or 2.')
#
#     return meta_batch_loss