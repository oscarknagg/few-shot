import torch
from torch.utils.data import Dataset

from few_shot.few_shot import create_nshot_task_label
from few_shot.metrics import categorical_accuracy


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model, optimiser, loss_fn, x, y, **kwargs):
    # Unpack arguments
    n, k, q = kwargs['n_shot'], kwargs['k_way'], kwargs['q_queries']
    order = kwargs['order']
    device = kwargs['device']
    inner_train_steps = kwargs['inner_train_steps']
    num_input_channels, height, width = x.shape[2:]

    task_gradients = []
    task_losses = []
    task_predictions = []
    for meta_batch in x:
        # By construction x is a 5D tensor of shape:
        # meta_batch_size, n*k + q*k, num_input_channels, width, height)
        # Hence when we iterate over the first, meta-batch, dimension we are iterating
        # x, _ = meta_batch
        # x = x.to(device, dtype=torch.double)
        x_task_train = meta_batch[:n * k] #.to(device)
        x_task_val = meta_batch[n * k:]

        # Create a fast model using the current meta model weights
        fast_model = model.__class__(num_input_channels, k).to(device, dtype=torch.double)
        copy_weights(from_model=model, to_model=fast_model)
        fast_opt = torch.optim.SGD(fast_model.parameters(), lr=kwargs['inner_lr'])

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Get batch
            y = create_nshot_task_label(k, n).to(device)

            # Perform update of model weights
            fast_model.train()
            fast_opt.zero_grad()
            logits = fast_model(x_task_train)
            loss = loss_fn(logits, y)
            loss.backward()
            fast_opt.step()

        # Do a pass of the model on the validation data from the current task
        logits = fast_model(x_task_val)
        y = create_nshot_task_label(k, q).to(device)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        grads = torch.autograd.grad(loss, fast_model.parameters(), create_graph=True)
        named_grads = {name: g for ((name, _), g) in zip(fast_model.named_parameters(), grads)}
        task_gradients.append(named_grads)

        del fast_model

    sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).sum(dim=0)
                          for k in task_gradients[0].keys()}

    if kwargs['train']:
        if order == 1:
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros(k, num_input_channels, height, width).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()
        else:
            raise NotImplementedError

    return torch.stack(task_losses).mean().item(), torch.cat(task_predictions)


def copy_weights(from_model, to_model):
    """Copies the weights from one model to another model."""
    # TODO: won't copy buffers, e.g. for batch norm

    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def weight_differences(from_model, to_model):
    weight_diff = 0
    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        if is_linear or is_conv:
            weight_diff += torch.abs(m_to.weight - m_from.weight).mean()

    return weight_diff

