import torch

from few_shot.few_shot import create_nshot_task_label


# def meta_gradient_step(model, optimiser, loss_fn, x, y, **kwargs):
#     task_gradients = []
#     task_losses = []
#     for meta_batch in range(meta_batch_size):
#         x, _ = taskloader.__iter__().next()
#
#         y = create_nshot_task_label(kwargs['k_way'], kwargs['n_shot'])
#
#         model.train()
#         logits = model(x)
#         loss = loss_fn(logits, y)
#         loss.backward(retain_graph=True)
#
#         task_losses.append(loss)
#
#         grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#         named_grads = {name: g for ((name, _), g) in zip(model.named_parameters(), grads)}
#
#         task_gradients.append(named_grads)
#
#     sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).sum(dim=0)
#                           for k in task_gradients[0].keys()}


def copy_weights(from_model, to_model):
    ''' Set this module's weights to be the same as those of 'net' '''
    # TODO: breaks if nets are not identical
    # TODO: won't copy buffers, e.g. for batch norm
    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()
