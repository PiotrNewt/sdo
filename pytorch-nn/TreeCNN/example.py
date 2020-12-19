import numpy as np
import torch.nn as nn
import tcnn
from util import prepare_trees

tree1 = (
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    ((0, 1, 0, 0, 0, 0, 0, 0, 1, 0), ((0, 0, 0, 0, 1, 0, 0, 1, 0, 0),), ((0, 1, 0, 0, 0, 0, 0, 0, 0, 1),)),
    ((0, 0, 0, 0, 0, 0, 0, 0, 1, 0), ((0, 0, 1, 0, 0, 0, 0, 0, 0, 0),), ((0, 0, 0, 0, 0, 1, 0, 0, 1, 1),))
)

trees = [tree1]

def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[1]

def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]

def transformer(x):
    return np.array(x[0])

net = nn.Sequential(
    tcnn.BinaryTreeConv(10, 16),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(16, 8),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(8, 4),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.DynamicPooling()
)

prepared_trees = prepare_trees(trees, transformer, left_child, right_child)
print(net(prepared_trees))