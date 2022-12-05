from typing import Optional, Tuple, Union

import torch
from torch import Tensor

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from torch_geometric.utils import degree, sort_edge_index, subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

classes = __all__ = [
    "drop_edge",
    "drop_node",
    "drop_path",
    "add_random_walk_edge",
]


def drop_edge(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    p: float = 0.5,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""DropEdge: Sampling edge using a uniform distribution
    from the `"DropEdge: Towards Deep Graph Convolutional
    Networks on Node Classification" <https://arxiv.org/abs/1907.10903>`_
    paper (ICLR'20)

    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each edge, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from mooon import drop_edge
        edge_index = torch.tensor([[1, 2], [3,4]])
        drop_edge(edge_index, p=0.5)

    See also
    --------
    :class:`greatx.functional.DropEdge`

    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= p
    edge_index = edge_index[:, edge_mask]
    if edge_weight is not None:
        edge_weight = edge_weight[edge_mask]
    return edge_index, edge_weight


def drop_node(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    p: float = 0.5,
    training: bool = True,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """DropNode: Sampling node using a uniform distribution
    from the `"Graph Contrastive Learning
    with Augmentations" <https://arxiv.org/abs/2010.139023>`_
    paper (NeurIPS'20)

    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
    p : float, optional
        the probability of dropping out on each node, by default 0.5
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from mooon import drop_node
        edge_index = torch.tensor([[1, 2], [3,4]])
        drop_node(edge_index, p=0.5)

    See also
    --------
    :class:`greatx.functional.DropNode`

    """

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or not p:
        return edge_index, edge_weight

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    return subgraph(node_mask, edge_index, edge_weight)


def drop_path(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    p: float = 0.5,
    walks_per_node: int = 1,
    walk_length: int = 3,
    num_nodes: Optional[int] = None,
    start: Union[str, Tensor] = 'node',
    is_sorted: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """DropPath: a structured form of :class:`~mooon.drop_edge`
    from the `"MaskGAE: Masked Graph Modeling Meets
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_
    paper (arXiv'22)


    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
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
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from mooon import drop_path
        edge_index = torch.tensor([[1, 2], [3,4]])
        drop_path(edge_index, p=0.5)

        # specify root nodes
        drop_path(edge_index, start=torch.tensor([1,2]))


    See also
    --------
    :class:`greatx.functional.DropPath`
    """

    if torch_cluster is None:
        raise ImportError("`torch_cluster` is not installed.")

    if not training:
        return edge_index, edge_weight

    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    assert isinstance(start, Tensor) or start in ['node', 'edge']
    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)

    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`drop_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not is_sorted:
        edge_index = sort_edge_index(edge_index, edge_weight,
                                     num_nodes=num_nodes)
        if edge_weight is not None:
            edge_index, edge_weight = edge_index

    row, col = edge_index

    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
        start = row[sample_mask].repeat(walks_per_node)
    elif start == 'node':
        perm = torch.randperm(num_nodes, device=edge_index.device)
        start = perm[:round(num_nodes * p)].repeat(walks_per_node)
    elif start.dtype == torch.bool:
        start = start.nonzero().view(-1)

    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    edge_mask[e_id] = False

    if edge_weight is not None:
        edge_weight = edge_weight[edge_mask]

    return edge_index[:, edge_mask], edge_weight


def add_random_walk_edge(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    start: Optional[Tensor] = None,
    walks_per_node: int = 1,
    walk_length: int = 3,
    skip_first: bool = True,
    num_nodes: Optional[int] = None,
    is_sorted: bool = False,
    training: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Adds edges and corresponding edge weights based on
    random walks.


    Parameters
    ----------
    edge_index : torch.Tensor
        the input edge index
    edge_weight : Optional[Tensor], optional
        the input edge weight, by default None
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
    training : bool, optional
        whether the model is during training,
        do nothing if :obj:`training=True`, by default True

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

        from mooon import add_random_walk_edge
        edge_index = torch.tensor([[1, 2], [3,4]])
        add_random_walk_edge(edge_index)

        # specify root nodes
        add_random_walk_edge(edge_index, start=torch.tensor([1,2]))

    See also
    --------
    :class:`greatx.functional.AddRandomWalkEdge`
    """

    if random_walk is None:
        raise ImportError('`add_random_walk_edge` requires `torch-cluster`.')

    if not training:
        return edge_index, edge_weight

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not is_sorted:
        edge_index = sort_edge_index(edge_index, edge_weight,
                                     num_nodes=num_nodes)
        if edge_weight is not None:
            edge_index, edge_weight = edge_index

    row, col = edge_index
    device = edge_index.device

    if start is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        start = torch.arange(num_nodes, device=device)
    elif start.dtype == torch.bool:
        start = start.nonzero().view(-1)

    start = start.repeat(walks_per_node)
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    p = q = 1.0
    walks = random_walk(rowptr, col, start, walk_length, p, q)[0]

    if skip_first:
        assert walk_length > 1
        rw_row = walks[:, [0]].repeat(1, walk_length - 1)
        rw_col = walks[:, 2:]
    else:
        rw_row = walks[:, [0]].repeat(1, walk_length)
        rw_col = walks[:, 1:]

    aug_edge_index = torch.stack([rw_row, rw_col]).view(2, -1).contiguous()
    # filter self-loops
    mask = aug_edge_index[0] != aug_edge_index[1]
    aug_edge_index = aug_edge_index[:, mask]
    edge_index = torch.cat([edge_index, aug_edge_index], dim=1)

    if edge_weight is not None:
        assert edge_weight.ndim == 1
        aug_edge_weight = 1. / torch.arange(
            int(skip_first) + 1, walk_length + 1, dtype=torch.float,
            device=device)
        aug_edge_weight = aug_edge_weight.repeat(start.size(0))[mask]
        edge_weight = torch.cat([edge_weight, aug_edge_weight])

    return edge_index, edge_weight
