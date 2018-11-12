from torch.utils.data import Dataset, Sampler
from typing import List, Iterable
import numpy as np
import torch

from few_shot.metrics import categorical_accuracy
from few_shot.callbacks import Callback


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
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

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

    def __init__(self, eval_fn, num_tasks, n_shot, k_way, q_queries, taskloader, prepare_batch, prefix='val_', **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
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
