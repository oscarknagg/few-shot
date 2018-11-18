import unittest

import torch
from torch.utils.data import DataLoader

from few_shot.core import NShotTaskSampler
from few_shot.datasets import DummyDataset
from few_shot.matching import matching_net_predictions
from few_shot.utils import pairwise_distances


class TestMatchingNets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset(samples_per_class=1000, n_classes=20)

    def _test_n_k_q_combination(self, n, k, q):
        n_shot_taskloader = DataLoader(self.dataset,
                                       batch_sampler=NShotTaskSampler(self.dataset, 100, n, k, q))

        # Load a single n-shot, k-way task
        for batch in n_shot_taskloader:
            x, y = batch
            break

        # Take just dummy label features and a little bit of noise
        # So distances are never 0
        support = x[:n * k, 1:]
        queries = x[n * k:, 1:]
        support += torch.rand_like(support)
        queries += torch.rand_like(queries)

        distances = pairwise_distances(queries, support, 'cosine')

        # Calculate "attention" as softmax over distances
        attention = (-distances).softmax(dim=1).cuda()

        y_pred = matching_net_predictions(attention, n, k, q)

        self.assertEqual(
            y_pred.shape,
            (q * k, k),
            'Matching Network predictions must have shape (q * k, k).'
        )

        y_pred_sum = y_pred.sum(dim=1)
        self.assertTrue(
            torch.all(
                torch.isclose(y_pred_sum, torch.ones_like(y_pred_sum).double())
            ),
            'Matching Network predictions probabilities must sum to 1 for each '
            'query sample.'
        )

    def test_matching_net_predictions(self):
        test_combinations = [
            (1, 5, 5),
            (5, 5, 5),
            (1, 20, 5),
            (5, 20, 5)
        ]

        for n, k, q in test_combinations:
            self._test_n_k_q_combination(n, k, q)