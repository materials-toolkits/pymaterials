import torch

from materials_toolkit.graph.fully_connected import fully_connected

from .utils import order_edges


def test_fully_collected_self_loop_directed():
    num_nodes = torch.tensor([3, 4, 2], dtype=torch.long)
    edges = fully_connected(num_nodes, self_loop=True, directed=True)
    edges_default = fully_connected(num_nodes)

    edges = order_edges(edges)
    edges_default = order_edges(edges_default)

    assert (edges == edges_default).all()
    assert (
        edges
        == torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    8,
                    8,
                ],
                [
                    0,
                    1,
                    2,
                    0,
                    1,
                    2,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    7,
                    8,
                ],
            ]
        )
    ).all()


def test_fully_collected_self_loop():
    num_nodes = torch.tensor([3, 4, 2], dtype=torch.long)
    edges = fully_connected(num_nodes, self_loop=True, directed=False)

    edges = order_edges(edges)

    assert (
        edges
        == torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    6,
                    7,
                    7,
                    8,
                ],
                [
                    0,
                    1,
                    2,
                    1,
                    2,
                    2,
                    3,
                    4,
                    5,
                    6,
                    4,
                    5,
                    6,
                    5,
                    6,
                    6,
                    7,
                    8,
                    8,
                ],
            ]
        )
    ).all()


def test_fully_collected_directed():
    num_nodes = torch.tensor([3, 4, 2], dtype=torch.long)
    edges = fully_connected(num_nodes, self_loop=False, directed=True)

    edges = order_edges(edges)

    assert (
        edges
        == torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8],
                [1, 2, 0, 2, 0, 1, 4, 5, 6, 3, 5, 6, 3, 4, 6, 3, 4, 5, 8, 7],
            ]
        )
    ).all()


def test_fully_collected_undirected():
    num_nodes = torch.tensor([3, 4, 2], dtype=torch.long)
    edges = fully_connected(num_nodes, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (
        edges
        == torch.tensor(
            [
                [0, 0, 1, 3, 3, 3, 4, 4, 5, 7],
                [1, 2, 2, 4, 5, 6, 5, 6, 6, 8],
            ]
        )
    ).all()


def test_fully_collected_masked():
    num_nodes = torch.tensor([3, 4, 2], dtype=torch.long)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[0, 0, 1, 7], [1, 2, 2, 8]])).all()

    mask = torch.tensor([False, True, True], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[3, 3, 3, 4, 4, 5, 7], [4, 5, 6, 5, 6, 6, 8]])).all()

    mask = torch.tensor([False, False, True], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[7], [8]])).all()

    mask = torch.tensor([False, True, True], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[3, 3, 3, 4, 4, 5, 7], [4, 5, 6, 5, 6, 6, 8]])).all()

    mask = torch.tensor([True, False, False], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[0, 0, 1], [1, 2, 2]])).all()

    mask = torch.tensor([False, True, False], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[3, 3, 3, 4, 4, 5], [4, 5, 6, 5, 6, 6]])).all()

    mask = torch.tensor([False, False, False], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[], []])).all()

    mask = torch.tensor([False, True, False], dtype=torch.bool)
    edges = fully_connected(num_nodes, mask=mask, self_loop=False, directed=False)

    edges = order_edges(edges)

    assert (edges == torch.tensor([[3, 3, 3, 4, 4, 5], [4, 5, 6, 5, 6, 6]])).all()
