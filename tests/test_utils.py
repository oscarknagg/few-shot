import unittest
import numpy as np
import torch
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from few_shot.utils import *
from config import PATH


class TestDistance(unittest.TestCase):
    def test_query_support_distances(self):
        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two

        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([0, 0])
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 0])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([1, 2])
        support[2] = torch.Tensor([2, 2])

        distances = pairwise_distances(query, support, 'l2')
        self.assertEqual(
            distances.shape, (q * k, k),
            'Output should have shape (q * k, k).'
        )

        # Calculate squared distances by iterating through all query-support pairs
        for i, q_ in enumerate(query):
            for j, s_ in enumerate(support):
                self.assertEqual(
                    (q_ - s_).pow(2).sum(),
                    distances[i, j].item(),
                    'The jth column of the ith row should be the squared distance between the '
                    'ith query sample and the kth query sample'
                )

        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two
        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([1, 0])
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 1])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([-1, -1])
        support[2] = torch.Tensor([0, 2])

        distances = pairwise_distances(query, support, 'cosine')

        # Calculate distances by iterating through all query-support pairs
        for i, q_ in enumerate(query):
            for j, s_ in enumerate(support):
                self.assertTrue(
                    torch.isclose(1-CosineSimilarity(dim=0)(q_, s_), distances[i, j], atol=2e-8),
                    'The jth column of the ith row should be the squared distance between the '
                    'ith query sample and the kth query sample'
                )

    def test_no_nans_on_zero_vectors(self):
        """Cosine distance calculation involves a divide-through by vector magnitude which
        can divide by zeros to occur.
        """
        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two
        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([0, 0])  # First query sample is all zeros
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 1])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([-1, -1])
        support[2] = torch.Tensor([0, 0])  # Third support sample is all zeros

        distances = pairwise_distances(query, support, 'cosine')

        self.assertTrue(torch.isnan(distances).sum() == 0, 'Cosine distances between 0-vectors should not be nan')


class TestAutogradGraphRetrieval(unittest.TestCase):
    def test_retrieval(self):
        """Create a simple autograd graph and check that the output is what is expected"""
        x = torch.ones(2, 2, requires_grad=True)
        y = x + 2
        # The operation on the next line will create two edges because the y variable is
        # y variable is used twice.
        z = y * y
        out = z.mean()

        nodes, edges = autograd_graph(out)

        # This is quite a brittle test as it will break if the names of the autograd Functions change
        # TODO: Less brittle test

        expected_nodes = [
            'MeanBackward1',
            'MulBackward0',
            'AddBackward0',
            'AccumulateGrad',
        ]

        self.assertEqual(
            set(expected_nodes),
            set(n.__class__.__name__ for n in nodes),
            'autograd_graph() must return all nodes in the autograd graph.'
        )

        # Check for the existence of the expected edges
        expected_edges = [
            ('MulBackward0', 'MeanBackward1'),  # z = y * y, out = z.mean()
            ('AddBackward0', 'MulBackward0'),    # y = x + 2, z = y * y
            ('AccumulateGrad', 'AddBackward0'),   # x = torch.ones(2, 2, requires_grad=True), y = x + 2
        ]
        for e in edges:
            self.assertIn(
                (e[0].__class__.__name__, e[1].__class__.__name__),
                expected_edges,
                'autograd_graph() must return all edges in the autograd graph.'
            )

        # Check for two edges between the AddBackward node and the ThMulBackward node
        num_y_squared_edges = 0
        for e in edges:
            if e[0].__class__.__name__ == 'AddBackward0' and e[1].__class__.__name__ == 'MulBackward0':
                num_y_squared_edges += 1

        self.assertEqual(
            num_y_squared_edges,
            2,
            'autograd_graph() must return multiple edges between nodes if they exist.'
        )
