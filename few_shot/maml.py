import torch

from few_shot.few_shot import create_nshot_task_label


def meta_gradient_step(model, optimiser, loss_fn, x, y, **kwargs):
    # Unpack arguments
    model, meta_model = model
    inner_optimiser, meta_optimiser = optimiser

    task_gradients = []
    task_losses = []
    for meta_batch in range(kwargs['meta_batch_size']):
        print('META BATCH: ', meta_batch)
        # Get all batches using NShotSampler.
        # The 'support set' will be uesd to train the model and the 'query set'
        # will be used to calculate meta gradients
        x, _ = kwargs['taskloader'].__iter__().next()
        x = x.to(kwargs['device'], dtype=torch.double)

        x_task_train = x[:kwargs['n'] * kwargs['k'] * kwargs['inner_train_steps']].to(kwargs['device'])
        x_task_val = x[kwargs['n'] * kwargs['k'] * kwargs['inner_train_steps']:]

        # Recreate the fast model using the current meta model weights
        copy_weights(from_model=meta_model, to_model=model)

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(kwargs['inner_train_steps']):
            print('- Inner Batch: ', inner_batch)
            # Get batch
            y = create_nshot_task_label(kwargs['k'], kwargs['n']).to(kwargs['device'])

            # Perform update of model weights
            model.train()
            inner_optimiser.zero_grad()
            logits = model(x_task_train)
            loss = loss_fn(logits, y)
            loss.backward()
            inner_optimiser.step()

        # Do a pass of the model on the validation data from the current task
        logits = model(x_task_val)
        y = create_nshot_task_label(kwargs['k'], kwargs['n']).to(kwargs['device'])
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Accumulate losses and gradients
        task_losses.append(loss)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        named_grads = {name: g for ((name, _), g) in zip(model.named_parameters(), grads)}
        task_gradients.append(named_grads)

    # sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).sum(dim=0)
    #                       for k in task_gradients[0].keys()}

    if kwargs['order'] == 2:
        summed_task_losses = torch.stack(task_losses).mean()

        meta_optimiser.zero_grad()
        summed_task_losses.backward()
        meta_optimiser.step()

    return summed_task_losses


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

