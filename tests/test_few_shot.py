import unittest

import torch
from torch.utils.data import DataLoader

from few_shot.core import create_nshot_task_label, NShotTaskSampler
from few_shot.datasets import DummyDataset


class TestNShotLabel(unittest.TestCase):
    def test_label(self):
        n = 1
        k = 5
        q = 1

        y = create_nshot_task_label(k, q)


class TestNShotSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DummyDataset(samples_per_class=1000, n_classes=20)

    def test_n_shot_sampler(self):
        n, k, q = 2, 4, 3
        n_shot_taskloader = DataLoader(self.dataset,
                                       batch_sampler=NShotTaskSampler(self.dataset, 100, n, k, q))

        # Load a single n-shot task and check it's properties
        for x, y in n_shot_taskloader:
            support = x[:n*k]
            queries = x[n*k:]
            support_labels = y[:n*k]
            query_labels = y[n*k:]

            # Check ordering of support labels is correct
            for i in range(0, n * k, n):
                support_set_labels_correct = torch.all(support_labels[i:i + n] == support_labels[i])
                self.assertTrue(
                    support_set_labels_correct,
                    'Classes of support set samples should be arranged like: '
                    '[class_1]*n + [class_2]*n + ... + [class_k]*n'
                )

            # Check ordering of query labels is correct
            for i in range(0, q * k, q):
                support_set_labels_correct = torch.all(query_labels[i:i + q] == query_labels[i])
                self.assertTrue(
                    support_set_labels_correct,
                    'Classes of query set samples should be arranged like: '
                    '[class_1]*q + [class_2]*q + ... + [class_k]*q'
                )

            # Check labels are consistent across query and support
            for i in range(k):
                self.assertEqual(
                    support_labels[i*n],
                    query_labels[i*q],
                    'Classes of query and support set should be consistent.'
                )

            # Check no overlap of IDs between support and query.
            # By construction the first feature in the DummyDataset is the
            # id of the sample in the dataset so we can use this to test
            # for overlap betwen query and suppport samples
            self.assertEqual(
                len(set(support[:, 0].numpy()).intersection(set(queries[:, 0].numpy()))),
                0,
                'There should be no overlap between support and query set samples.'
            )

            break