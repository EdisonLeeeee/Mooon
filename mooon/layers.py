from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mooon.functional import (add_random_walk_edge, drop_edge, drop_node,
                              drop_path)

classes = __all__ = [
    "DropEdge",
    "DropNode",
    "DropPath",
    "AddRandomWalkEdge",
]


class DropEdge(nn.Module):
    """DropEdge: Sampling edge using a uniform distribution
    from the `"DropEdge: Towards Deep Graph Convolutional
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each edge, by default 0.5

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Raises
    ------
    ValueError
        p is out of range [0,1]

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropEdge
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropEdge(p=0.5)(edge_index)

    See also
    --------
    :class:`greatx.functional.drop_edge`
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_edge(edge_index, edge_weight, self.p,
                         training=self.training)


class DropNode(nn.Module):
    """DropNode: Sampling node using a uniform distribution
    from the `"Graph Contrastive Learning
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    p : float, optional
        the probability of dropping out on each node, by default 0.5

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropNode
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropNode(p=0.5)(edge_index)

    See also
    --------
    :class:`greatx.functional.drop_node`
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_node(edge_index, edge_weight, self.p,
                         training=self.training)


class DropPath(nn.Module):
    """DropPath: a structured form of :class:`greatx.functional.drop_edge`
    from the `"MaskGAE: Masked Graph Modeling Meets
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)

    Parameters
    ----------
    p : float, optional
        the percentage of nodes in the graph that chosen as root nodes to
        perform random walks. By default, :obj:`p=0.5`.
    walks_per_node : int, optional
        number of walks per node, by default 1
    walk_length : int, optional
        number of walk length per node, by default 3
    num_nodes : int, optional
        number of total nodes in the graph, by default None
    start : Union[str, Tensor], optional
        the type of starting node chosen from "node", "edge",
        or custom nodes,  by default 'node'
    is_sorted : bool, optional
        whether the input :obj:`edge_index` is sorted

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Raises
    ------
    ImportError
        if :class:`torch_cluster` is not installed.
    ValueError
        :obj:`p` is out of scope [0,1]
    ValueError
        :obj:`p` is not integer value or a Tensor

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import DropPath
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        DropPath(p=0.5)(edge_index)

        # specify root nodes
        DropPath(start=torch.tensor([1,2]))(edge_index)

    See also
    --------
    :class:`greatx.functional.drop_path`
    """
    def __init__(self, p: float = 0.5, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 start: Union[str, Tensor] = 'node', is_sorted: bool = False):
        super().__init__()

        if isinstance(start, Tensor) and start.dtype == torch.bool:
            start = start.nonzero().view(-1)

        self.p = p
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.num_nodes = num_nodes
        self.start = start
        self.is_sorted = is_sorted

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return drop_path(edge_index, edge_weight, p=self.p,
                         walks_per_node=self.walks_per_node,
                         walk_length=self.walk_length,
                         num_nodes=self.num_nodes, start=self.start,
                         is_sorted=self.is_sorted, training=self.training)


class AddRandomWalkEdge(nn.Module):
    """Adds edges and corresponding edge weights based on
    random walks.

    Parameters
    ----------
    start : Tensor, optional
        the starting node to perform random walks, if None,
        use all nodes in the graph as root nodes,
        by default None
    walks_per_node : int, optional
        number of walks per node, by default 1
    walk_length : int, optional
        number of walk length per node, by default 3
    skip_first : bool, optional
        whether to skip the first-hop node when
        adding edges between root nodes and
        nodes sampled from random walks, by default False
    num_nodes : int, optional
        number of total nodes in the graph, by default None
    is_sorted : bool, optional
        whether the input :obj:`edge_index` is sorted

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        the output edge index and edge weight

    Raises
    ------
    ImportError
        if :class:`torch_cluster` is not installed.

    Example
    -------
    .. code-block:: python

        from greatx.nn.layers import AddRandomWalkEdge
        edge_index = torch.LongTensor([[1, 2], [3,4]])
        AddRandomWalkEdge()(edge_index)

        # specify root nodes
        AddRandomWalkEdge(start=torch.tensor([1,2]))(edge_index)

    See also
    --------
    :class:`greatx.functional.add_random_walk_edge`
    """
    def __init__(self, start: Optional[Tensor] = None, walks_per_node: int = 1,
                 walk_length: int = 3, skip_first: bool = True,
                 num_nodes: Optional[int] = None, is_sorted: bool = False):
        super().__init__()

        if isinstance(start, Tensor) and start.dtype == torch.bool:
            start = start.nonzero().view(-1)

        self.start = start
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.num_nodes = num_nodes
        self.is_sorted = is_sorted
        self.skip_first = skip_first

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """"""
        return add_random_walk_edge(
            edge_index, edge_weight, start=self.start,
            walks_per_node=self.walks_per_node, walk_length=self.walk_length,
            num_nodes=self.num_nodes, skip_first=self.skip_first,
            is_sorted=self.is_sorted, training=self.training)
