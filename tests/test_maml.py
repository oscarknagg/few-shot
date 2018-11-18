import unittest
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Tuple

from few_shot.datasets import DummyDataset
from few_shot.core import NShotTaskSampler, create_nshot_task_label
from few_shot.maml import meta_gradient_step
from few_shot.utils import autograd_graph


class DummyModel(torch.nn.Module):
    """Dummy 1 layer (0 hidden layer) model for testing purposes"""
    def __init__(self, k: int):
        super(DummyModel, self).__init__()
        self.out = torch.nn.Linear(2, k, bias=False)

    def forward(self, x):
        x = self.out(x)
        return x

    def functional_forward(self, x, weights):
        x = F.linear(x, weights['out.weight'])
        return x


class TestMAML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 1
        cls.k = 5
        cls.q = 1

        cls.meta_batch_size = 1

        cls.dummy = DummyDataset()
        cls.dummy_tasks = DataLoader(
            cls.dummy, batch_sampler=NShotTaskSampler(cls.dummy, cls.meta_batch_size, n=cls.n, k=cls.k, q=cls.q, num_tasks=1),
        )

        cls.model = DummyModel(cls.k).double()
        cls.opt = torch.optim.Adam(cls.model.parameters(), lr=0.001)

    def _get_maml_graph(self, order: int, inner_train_steps: int) -> Tuple[
            List[torch.autograd.Function],
            List[Tuple[torch.autograd.Function, torch.autograd.Function]]
        ]:
        """Gets the autograd graph for a single iteration of MAML.

        # Arguments:
            order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
            inner_train_steps: Number of gradient steps to fit the fast weights during each inner update

        # Returns
            nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
            edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
        """
        x, _ = self.dummy_tasks.__iter__().__next__()
        x = x.double().reshape(self.meta_batch_size, self.n * self.k + self.q * self.k, x.shape[-1])
        y = create_nshot_task_label(self.k, self.q).repeat(self.meta_batch_size)

        loss, y_pred = meta_gradient_step(self.model, self.opt, torch.nn.CrossEntropyLoss(), x, y, self.n, self.k,
                                          self.q, order=order, inner_train_steps=inner_train_steps, inner_lr=0.1,
                                          train=True, device='cpu')

        nodes, edges = autograd_graph(loss)

        return nodes, edges

    def _count_named_nodes(self, nodes: List[torch.autograd.Function], name: str) -> int:
        """Counts the number of autograd graph nodes that are named in a particular way

        # Arguments
            nodes: List of autograd Functions
            name: Name of Function to look for

        # Returns
            count: Number of s in `nodes` that match `name`
        """
        count = 0

        for n in nodes:
            if n.__class__.__name__ == name:
                count += 1

        return count

    def test_first_order(self):
        """Test the 1st order MAML is only perform a first order update by inspecting the autograd graph."""
        # Run a single meta batch using 1st order MAML
        nodes_1_1, _ = self._get_maml_graph(order=1, inner_train_steps=1)

        self.assertEqual(
            self._count_named_nodes(nodes_1_1, 'NllLossBackwardBackward'),
            0,
            'The autograd graph of 1st order MAML should not contain any double backwards operations'
        )

    def test_second_order(self):
        """Test that 2nd order MAML is genuinely performing a second order update by inspecting the autograd graph."""
        # Run a single meta batch using 1st order MAML
        nodes_1_1, _ = self._get_maml_graph(order=1, inner_train_steps=1)

        # Run a single meta batch using 2nd order MAML
        nodes_2_1, _ = self._get_maml_graph(order=2, inner_train_steps=1)

        # Run a single meta batch using 2nd order MAML
        nodes_2_2, _ = self._get_maml_graph(order=2, inner_train_steps=2)

        self.assertGreater(
            len(nodes_2_1),
            len(nodes_1_1),
            '2nd order MAML should produce a larger autograd graph than 1st order MAMl'
        )

        self.assertEqual(
            self._count_named_nodes(nodes_2_1, 'NllLossBackwardBackward'),
            1,
            '2nd order MAML with 1 inner step should produce an autograd graph with a single double backwards '
            'operation corresponding to taking the gradient of the gradient due to the first inner step.'
        )

        self.assertEqual(
            self._count_named_nodes(nodes_2_2, 'NllLossBackwardBackward'),
            2,
            '2nd order MAML with 2 inner steps should produce an autograd graph with two double backwards '
            'operations corresponding to taking the second differential of the gradients of the 1st and 2nd inner step.'
        )
