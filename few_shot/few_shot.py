from torch.utils.data import Dataset, Sampler
from torch.nn.utils import clip_grad_norm_
from typing import List, Iterable
import numpy as np
import torch

from few_shot.metrics import categorical_accuracy
from few_shot.callbacks import Callback
from few_shot.utils import pairwise_distances
from config import EPSILON


class NShotWrapper(Dataset):
    """Wraps one of the two Dataset classes to create a new Dataset that returns n-shot, k-way, q-query tasks."""
    def __init__(self, dataset, epoch_length, n, k, q):
        self.dataset = dataset
        self.epoch_length = epoch_length
        self.n_shot = n
        self.k_way = k
        self.q_queries = q

    def __getitem__(self, item):
        """Get a single n-shot, k-way, q-query task."""
        # Select classes
        episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k_way, replace=False)
        df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
        batch = []
        labels = []

        support_k = {k: None for k in episode_classes}
        for k in episode_classes:
            # Select support examples
            support = df[df['class_id'] == k].sample(self.n_shot)
            support_k[k] = support

            for i, s in support.iterrows():
                x, y = self.dataset[s['id']]
                batch.append(x)
                labels.append(k)

        for k in episode_classes:
            query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q_queries)
            for i, q in query.iterrows():
                x, y = self.dataset[q['id']]
                batch.append(x)
                labels.append(k)

        return np.stack(batch), np.array(labels)

    def __len__(self):
        return self.epoch_length


class NShotTaskSampler(Sampler):
    def __init__(self, dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 fixed_tasks: List[Iterable[int]] = None):
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks
        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            if self.fixed_tasks is None:
                # Get random classes
                episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
            else:
                # Loop through classes in fixed_tasks
                episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                self.i_task += 1

            df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
            batch = []

            support_k = {k: None for k in episode_classes}
            for k in episode_classes:
                # Select support examples
                support = df[df['class_id'] == k].sample(self.n)
                support_k[k] = support

                for i, s in support.iterrows():
                    batch.append(s['id'])

            for k in episode_classes:
                query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                for i, q in query.iterrows():
                    batch.append(q['id'])

            yield np.stack(batch)


def proto_net_episode(model, optimiser, loss_fn, x, y, **kwargs):
    if kwargs['train']:
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
    support = embeddings[:kwargs['n_shot']*kwargs['k_way']]
    queries = embeddings[kwargs['n_shot']*kwargs['k_way']:]
    prototypes = compute_prototypes(support, kwargs['k_way'], kwargs['n_shot'])

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, prototypes, kwargs['distance'])

    # Calculate log p_{phi} (y = k | x)
    log_p_y = (-distances).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if kwargs['train']:
        # Take gradient step
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss.item(), y_pred


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.

    This very simple function is separated so it can be easily tested.

    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    return support.reshape(k, n, -1).mean(dim=1)


class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: proto_net_episode or matching_net_episode
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self, eval_fn, num_tasks, n_shot, k_way, q_queries, task_loader, prepare_batch, prefix='val_', **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = task_loader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimiser = self.params['optimiser']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(self.model, self.optimiser, self.loss_fn, x, y,
                                        n_shot=self.n_shot, k_way=self.k_way, q_queries=self.q_queries,
                                        train=False, **self.kwargs)

            seen += y_pred.shape[0]

            totals['loss'] += loss * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen


def matching_net_episode(model, optimiser, loss_fn, x, y, **kwargs):
    if kwargs['train']:
        # Zero gradients
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model.encoder(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:kwargs['n_shot'] * kwargs['k_way']]
    queries = embeddings[kwargs['n_shot'] * kwargs['k_way']:]

    # Optionally apply full context embeddings
    if kwargs['fce']:
        # LSTM requires input of shape (seq_len, batch, input_size). `support` is of
        # shape (k_way * n_shot, embedding_dim) and we want the LSTM to treat the
        # support set as a sequence so add a single dimension to transform support set
        # to the shape (k_way * n_shot, 1, embedding_dim) and then remove the batch dimension
        # afterwards

        # Calculate the fully conditional embedding, g, for support set samples as described
        # in appendix A.2 of the paper. g takes the form of a bidirectional LSTM with a
        # skip connection from inputs to outputs
        support, _, _ = model.g(support.unsqueeze(1))
        support = support.squeeze(1)

        # Calculate the fully conditional embedding, f, for the query set samples as described
        # in appendix A.1 of the paper.
        queries = model.f(support, queries)

    # Efficiently calculate distance between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, support, kwargs['distance'])

    # Calculate "attention" as softmax over support-query distances
    attention = (-distances).softmax(dim=1)

    # Calculate predictions as in equation (1) from Matching Networks
    # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
    y_pred = matching_net_predictions(attention, kwargs['n_shot'], kwargs['k_way'], kwargs['q_queries'])

    # Calculated loss with negative log likelihood
    # Clip predictions for numerical stability
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    loss = loss_fn(clipped_y_pred.log(), y)

    if kwargs['train']:
        # Backpropagate gradients
        loss.backward()
        # I found training to be quite unstable so I clip the norm
        # of the gradient to be at most 1
        clip_grad_norm_(model.parameters(), 1)
        # Take gradient step
        optimiser.step()
    else:
        pass

    return loss.item(), y_pred


def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int) -> torch.Tensor:
    """Calculates Matching Network predictions based on equation (1) of the paper.

    The predictions are the weighted sum of the labels of the support set where the
    weights are the "attentions" (i.e. softmax over query-support distances) pointing
    from the query set samples to the support set samples.

    # Arguments
        attention: torch.Tensor containing softmax over query-support distances.
            Should be of shape (q * k, k * n)
        n: Number of support set samples per class, n-shot
        k: Number of classes in the episode, k-way
        q: Number of query samples per-class
    """
    if attention.shape != (q * k, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k)

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label(k, n).unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    y_pred = torch.mm(attention, y_onehot.cuda().double())

    return y_pred


def prepare_nshot_task(n, k, q):
    def prepare_nshot_task_(batch):
        x, y = batch
        x = x.double().cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k, q).cuda()
        return x, y

    return prepare_nshot_task_


def create_nshot_task_label(k, q):
    return torch.arange(0, k, 1 / q).long()
