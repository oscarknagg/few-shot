import torch
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.few_shot import NShotTaskSampler, create_nshot_task_label, prepare_nshot_task, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import OmniglotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int)

args = parser.parse_args()

if args.dataset == 'omniglot':
    n_epochs = 300
    dataset_class = OmniglotDataset
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    raise NotImplementedError
else:
    raise(ValueError('Unsupported dataset'))

evaluation_episodes = 1
episodes_per_epoch = 50

param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
print(param_str)


###################
# Create datasets #
###################
background = OmniglotDataset('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.meta_batch_size, n=args.n, k=args.k, q=1),
    num_workers=4
)
evaluation = OmniglotDataset('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.meta_batch_size, n=args.n, k=args.k, q=1),
    num_workers=4
)


############
# Training #
############
print(f'Training MAML on {args.dataset}...')
meta_model = OmniglotClassifier(num_input_channels, args.k).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
loss_fn = nn.CrossEntropyLoss().to(device)


def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        x = x.double().cuda()
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_


callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=evaluation_episodes,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, 1, args.meta_batch_size),
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        num_input_channels=num_input_channels,
        device=device,
        order=args.order,
        task_loader=evaluation_taskloader
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/maml/{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc'
    ),
    CSVLogger(PATH + f'/logs/maml/{param_str}.csv'),
]


fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, 1, args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True, 'task_loader': background_taskloader,
                         'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                         'num_input_channels': num_input_channels, 'inner_lr': args.inner_lr},
)
