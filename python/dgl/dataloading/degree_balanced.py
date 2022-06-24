"""Degree balanced dataloader."""
# pylint: disable=bad-super-call
from typing import Generic
import functools

import torch
import numpy as np
from ..dataloading.dataloader import _TensorizedDatasetIter, DataLoader, TensorizedDataset


class DegreeBalancedDataloader(DataLoader):
    """Dataloader class that balances the degrees of each node minibatch.

    Instead of having a fixed number of seed nodes in each mini-batch, this dataloader
    tries to balance the total node degrees of each mini-batch, which is useful when
    node degree variance is causing unbalanced mini-batch workloads.

    Parameters
    ----------
    g : DGLGraph
    The input graph to sample from.
    nids : Tensor of dict[str, Tensor]
        Seed node IDs.
    sampler : dgl.dataloading.Sampler
        The subgraph sampler.
    max_node : int
        Maximum number of nodes in a batch.
    max_edge : int
        Maximum number of edges in a batch.
    **kwargs : keyword arguments
        Other keyword arguments passed to :class:`~dgl.dataloading.DataLoader`.

    Examples
    ---------
    To use DegreeBalancedDataloader on RedditDataset:

    >>> import torch
    >>> import dgl
    >>> from dgl.dataloading.degree_balanced import DegreeBalancedDataloader
    >>> from dgl.data import RedditDataset

    >>> data = RedditDataset(self_loop=True)
    >>> g = data[0].to("cuda")
    >>> nids = torch.arange(g.number_of_nodes()).to(g.device)
    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    >>> dataloader = DegreeBalancedDataloader(
    ...     g, nids, sampler, max_node=5000, max_edge=500000,
    ...     shuffle=False, device="cuda", num_workers=0)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     print(blocks)
    """
    def __init__(self, g, nids, sampler, max_node, max_edge, \
        device='cpu', shuffle=False, use_uva=False, num_workers=0):

        dataset = DegreeBalancedDataset(max_node, max_edge, g, nids, shuffle)
        super().__init__(g,
                         dataset,
                         sampler,
                         device=device,
                         use_uva=use_uva,
                         shuffle=shuffle,
                         drop_last=False,
                         use_prefetch_thread=False,
                         num_workers=num_workers)

    def modify_max_edge(self, max_edge):
        """Modify maximum edges.

        Parameters
        ----------
        max_edge : int
            The modified maximum number of edges.
        """
        self.dataset.max_edge = max_edge
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_edge = max_edge

    def modify_max_node(self, max_node):
        """Modify maximum nodes.

        Parameters
        ----------
        max_node : int
            The modified maximum number of nodes.
        """
        self.dataset.max_node = max_node
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.max_node = max_node

    def reset_batch_node(self, node_count):
        """Reset batch node.

        Parameters
        ----------
        node_count : int
            The number of nodes to be rollback.
        """
        if self.dataset.curr_iter is not None:
            self.dataset.curr_iter.index -= node_count

    def __setattr__(self, __name, __value):
        super(Generic, self).__setattr__(__name, __value)


class DegreeBalancedDataset(TensorizedDataset):
    """Degree balanced tensorized dataset extended from dgl.dataloading."""
    def __init__(self, max_node, max_edge, g, train_nids, shuffle):
        super().__init__(train_nids, max_node, drop_last=False, shuffle=shuffle)
        self.device = train_nids.device
        self.max_node = max_node
        self.max_edge = max_edge

        # We change the shuffle stretegy here. Since we need to compute the prefix sum of
        # in degrees array.
        self._indices = torch.arange(train_nids.shape[0], dtype=torch.int64).share_memory_()
        if shuffle:
            np.random.shuffle(self._indices.numpy())
        # move __iter__ to here.
        # TODO: support multi processing
        id_tensor = self._id_tensor[self._indices.to(self._device)]

        # Compute the prefix sum in degrees for the graph.
        in_degrees = g.in_degrees(id_tensor.to(g.device)).cpu()
        prefix_sum_in_degrees = [0]
        prefix_sum_in_degrees.extend(np.cumsum(in_degrees).tolist())
        prefix_sum_in_degrees.append(2e18)

        self.curr_iter = DegreeBalancedDatasetIter(id_tensor, self.max_node, self.max_edge,
            prefix_sum_in_degrees, self._mapping_keys)

    def __getattr__(self, attribute_name):
        if attribute_name in DegreeBalancedDataset.functions:
            function = functools.partial(DegreeBalancedDataset.functions[attribute_name], self)
            return function
        else:
            return super(Generic, self).__getattr__(attribute_name)

    def shuffle(self):
        """We use another shuffle stretegy here."""
        return

    def __iter__(self):
        return self.curr_iter


class DegreeBalancedDatasetIter(_TensorizedDatasetIter):
    """Degree balanced tensorized datasetIter."""
    def __init__(self, dataset, max_node, max_edge, prefix_sum_in_degrees, mapping_keys):
        super().__init__(dataset, max_node,
            drop_last=False, mapping_keys=mapping_keys, shuffle=False)
        self.max_node = max_node
        self.max_edge = max_edge
        self.prefix_sum_in_degrees = prefix_sum_in_degrees
        self.num_item = self.dataset.shape[0]

    def get_end_idx(self):
        """Get end index by binary search."""
        binary_start = self.index + 1
        binary_end = min(self.index + self.max_node, self.num_item)
        if self.prefix_sum_in_degrees[binary_end] - self.prefix_sum_in_degrees[self.index] \
            <= self.max_edge:
            return binary_end
        binary_middle = 0
        while binary_end - binary_start > 1:
            binary_middle = (binary_start + binary_end) // 2
            if self.prefix_sum_in_degrees[binary_middle] - self.prefix_sum_in_degrees[self.index] \
                <= self.max_edge:
                binary_start = binary_middle
            else:
                binary_end = binary_middle - 1
        return binary_start

    def _next_indices(self):
        """Overwrite next indices."""
        if self.index >= self.num_item:
            raise StopIteration
        end_idx = self.get_end_idx()
        batch = self.dataset[self.index:end_idx]
        self.index = end_idx
        return batch
