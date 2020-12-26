import torch
import torch.nn as nn
import torch.nn.functional as F
import network.tcnn as tcnn

# we need a encode model for plan encode.
# and we can put this encode model in QNetwork
# plan-tree --> tree-cnn --> pooling --> linear --> action

class TQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, tcnn_input_trees=1, tcnn_output=4, fc1_units=64, fc2_units=64):
        super(TQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.encode = nn.Sequential(
            tcnn.BinaryTreeConv(state_size, 16),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.BinaryTreeConv(16, 8),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.BinaryTreeConv(8, tcnn_output),
            tcnn.TreeLayerNorm(),
            tcnn.TreeActivation(nn.ReLU()),
            tcnn.DynamicPooling()
        )
        self.fc1 = nn.Linear(tcnn_output*tcnn_input_trees, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = self.encode(state).reshape(1,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
