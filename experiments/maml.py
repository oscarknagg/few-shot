import torch
from torch.utils.data import DataLoader
from torch import nn
import argparse
import numpy as np

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.few_shot import NShotTaskSampler, create_nshot_task_label
from few_shot.models import OmniglotClassifier
from few_shot.maml import copy_weights, weight_differences
from few_shot.metrics import categorical_accuracy

assert torch.cuda.is_available()
device = torch.device('cuda')


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.01, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=2, type=int)

args = parser.parse_args()

if args.inner_train_steps != 1:
    raise NotImplementedError

if args.dataset == 'omniglot':
    n_epochs = 120
    dataset_class = OmniglotDataset
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    raise NotImplementedError
else:
    raise(ValueError('Unsupported dataset'))

evaluation_episodes = 1000
episodes_per_epoch = 500

#########
# Model #
#########
meta_model = OmniglotClassifier(num_input_channels, args.k).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
# model = OmniglotClassifier(num_input_channels, args.k).to(device, dtype=torch.double)
# inner_optimiser = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
loss_fn = nn.CrossEntropyLoss().to(device)


###################
# Create datasets #
###################
background = OmniglotDataset('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, n=args.n * args.inner_train_steps, k=args.k, q=1),
    num_workers=4
)
evaluation = OmniglotDataset('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, n=args.n * args.inner_train_steps, k=args.k, q=1),
    num_workers=4
)

fixed_tasks = [
    (0, 10, 20, 30, 40),
    (50, 60, 70, 80, 90)
]
fixed_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(evaluation, 2,
                                   n=args.n * args.inner_train_steps,
                                   k=args.k,
                                   q=1,
                                   fixed_tasks=fixed_tasks),
    num_workers=4
)


from config import PATH
with open(PATH + '/logs/maml/test.csv', 'w') as f:
    print('loss,pre_acc,post_acc', file=f)

for i in range(1000):
    print(i)
    task_gradients = []
    task_losses = []
    task_accuracies_pre_update = []
    task_accuracies_post_update = []

    # for meta_batch in range(args.meta_batch_size):
    for meta_batch in fixed_taskloader:
        # print('META BATCH: ', meta_batch)
        # Get all batches using NShotSampler.
        # The 'support set' will be uesd to train the model and the 'query set'
        # will be used to calculate meta gradients
        x, _ = meta_batch
        x = x.to(device, dtype=torch.double)

        x_task_train = x[:args.n * args.k * args.inner_train_steps].to(device)
        x_task_val = x[args.n * args.k * args.inner_train_steps:]

        # Recreate the fast model using the current meta model weights
        # print('WEIGHT DIFF = ', weight_differences(meta_model, model))
        fast_model = OmniglotClassifier(num_input_channels, args.k).to(device, dtype=torch.double)
        copy_weights(from_model=meta_model, to_model=fast_model)
        fast_opt = torch.optim.SGD(fast_model.parameters(), lr=args.inner_lr)

        # Get pre-update accuracy
        # Do a pass of the model on the validation data from the current task
        fast_model.eval()
        logits = fast_model(x_task_val)
        y = create_nshot_task_label(args.k, args.n).to(device)
        y_pred = logits.softmax(dim=1)
        acc = categorical_accuracy(y, y_pred)
        task_accuracies_pre_update.append(acc)

        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(args.inner_train_steps):
            # print('- Inner Batch: ', inner_batch)
            # Get batch
            y = create_nshot_task_label(args.k, args.n).to(device)

            # Perform update of model weights
            fast_model.train()
            fast_opt.zero_grad()
            logits = fast_model(x_task_train)
            loss = loss_fn(logits, y)
            loss.backward()
            fast_opt.step()

        # Do a pass of the model on the validation data from the current task
        logits = fast_model(x_task_val)
        y = create_nshot_task_label(args.k, args.n).to(device)
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get accuracies
        y_pred = logits.softmax(dim=1)
        acc = categorical_accuracy(y, y_pred)
        task_accuracies_post_update.append(acc)

        # Accumulate losses and gradients
        task_losses.append(loss)
        grads = torch.autograd.grad(loss, fast_model.parameters(), create_graph=True)
        named_grads = {name: g for ((name, _), g) in zip(fast_model.named_parameters(), grads)}
        task_gradients.append(named_grads)

    summed_task_losses = torch.stack(task_losses).mean()
    sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).sum(dim=0)
                          for k in task_gradients[0].keys()}

    if args.order == 2:
        print('LOSS = ', summed_task_losses)
        print('PRE ACC = ', torch.Tensor(task_accuracies_pre_update).mean())
        print('POST ACC = ', torch.Tensor(task_accuracies_post_update).mean())
        meta_model.train()
        meta_optimiser.zero_grad()
        summed_task_losses.backward()
        meta_optimiser.step()
        with open(PATH + '/logs/maml/test.csv', 'a') as f:
            print(f'{summed_task_losses},'
                  f'{torch.Tensor(task_accuracies_pre_update).mean()},'
                  f'{torch.Tensor(task_accuracies_post_update).mean()}', file=f)
    elif args.order == 1:
        # Stolen from pytorch-maml on Github
        def replace_grad(parameter_name):
            def replace_grad_(module):
                return sum_task_gradients[parameter_name]

            return replace_grad_


        hooks = []
        for name, param in meta_model.named_parameters():
            hooks.append(
                param.register_hook(replace_grad(name))
            )

        meta_model.train()
        meta_optimiser.zero_grad()
        logits = meta_model(torch.zeros(args.k, 1, 28, 28).to(device, dtype=torch.double))
        loss = loss_fn(logits, create_nshot_task_label(args.k, args.n).to(device))
        loss.backward()
        meta_optimiser.step()

        print('LOSS = ', summed_task_losses)
        print('PRE ACC = ', torch.Tensor(task_accuracies_pre_update).mean())
        print('POST ACC = ', torch.Tensor(task_accuracies_post_update).mean())

        with open(PATH + '/logs/maml/test.csv',  'a') as f:
            print(f'{summed_task_losses},'
                  f'{torch.Tensor(task_accuracies_pre_update).mean()},'
                  f'{torch.Tensor(task_accuracies_post_update).mean()}', file=f)

        for h in hooks:
            h.remove()


