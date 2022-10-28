import torch
from torch_geometric.testing import withPackage

from mooon import drop_edge, drop_node, drop_path


def test_drop_node():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight = torch.tensor([1., 2., 3., 4., 5., 6])

    out = drop_node(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()

    torch.manual_seed(5)
    out = drop_node(edge_index)
    print(out)
    assert out[0].tolist() == [[2, 3], [3, 2]]
    assert out[1] is None

    torch.manual_seed(5)
    out = drop_node(edge_index, edge_weight)
    print(out)
    assert out[0].tolist() == [[2, 3], [3, 2]]
    assert out[1].tolist() == [
        5.,
        6.,
    ]


def test_drop_edge():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight = torch.tensor([1., 2., 3., 4., 5., 6])

    out = drop_edge(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1] is None

    torch.manual_seed(5)
    out = drop_edge(edge_index)
    assert out[0].tolist() == [[0, 1, 2, 2], [1, 2, 1, 3]]
    assert out[1] is None

    torch.manual_seed(5)
    out = drop_edge(edge_index, edge_weight)
    assert out[0].tolist() == [[0, 1, 2, 2], [1, 2, 1, 3]]
    assert out[1].tolist() == [1., 3., 4., 5.]


@withPackage('torch_cluster')
def test_drop_path():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight = torch.tensor([1., 2., 3., 4., 5., 6])

    out = drop_path(edge_index, training=False)
    assert edge_index.tolist() == out[0].tolist()
    assert out[1] is None

    torch.manual_seed(4)
    out = drop_path(edge_index, p=0.2)
    assert out[0].tolist() == [[1, 2, 3], [2, 3, 2]]
    assert out[1] is None

    torch.manual_seed(4)
    out = drop_path(edge_index, edge_weight, p=0.2)
    assert out[0].tolist() == [[1, 2, 3], [2, 3, 2]]
    assert out[1].tolist() == [3., 5., 6.]

    # test with unsorted edges
    torch.manual_seed(6)
    edge_index = torch.tensor([[3, 5, 2, 2, 2, 1], [1, 0, 0, 1, 3, 2]])
    out = drop_path(edge_index, p=0.2)
    assert out[0].tolist() == [[2, 3, 5], [0, 1, 0]]
    assert out[1] is None

    # test with isolated nodes
    torch.manual_seed(7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 2, 4]])
    out = drop_path(edge_index, p=0.2)
    assert out[0].tolist() == [[2, 3], [2, 4]]
    assert out[1] is None
