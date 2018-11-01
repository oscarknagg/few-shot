"""
Reproduce Matching Network results of Vinyals et al
"""
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.models import get_few_shot_encoder
from few_shot.few_shot import NShotSampler, prepare_nshot_task, matching_net_eposide, EvaluateFewShot
from few_shot.train import fit
from few_shot.callbacks import *
from config import PATH


assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--fce', type=lambda x: x.lower()[0] == 't')
parser.add_argument('--distance', default='cosine')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=15, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--lstm-size', default=64, type=int)
parser.add_argument('--lstm-layers', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
    lstm_input_size = 64
elif args.dataset == 'miniImageNet':
    n_epochs = 40
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 40
    lstm_input_size = 1600
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'matching_net_{args.dataset}_n={args.n_train}_k={args.k_train}_q={args.q_train}_dist={args.distance}_' \
            f'fce={args.fce}'


#############
# Functions #
#############
class BidrectionalLSTM(nn.Module):
    def __init__(self):
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = 1
        self.batch_size = 1
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            num_layers=args.lstm_layers,
                            hidden_size=args.lstm_size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


# class AttentionLSTM(nn.Module):
#     def __init__(self, unrolling_steps):
#         super(AttentionLSTM, self).__init__()


class MatchingNetwork(nn.Module):
    def __init__(self, n, k, q, fce, num_input_channels):
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM().to(device, dtype=torch.double)
            # self.f =

    def forward(self, inputs):
        pass


###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)


#########
# Model #
#########
model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels)
model.to(device, dtype=torch.double)


############
# Training #
############
print(f'Training Matching Network on {args.dataset}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=matching_net_eposide,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        task_loader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        fce=args.fce,
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/matching_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/matching_nets/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=matching_net_eposide,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'fce': args.fce, 'distance': args.distance}
)
