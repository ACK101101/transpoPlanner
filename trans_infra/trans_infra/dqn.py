import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """f(s) -> q : takes concat graph embeddings of states and predicts their q-vals"""
    def __init__(self, input_dim, hidden_dim, out_dim=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        print(f"    dqn x1: {x}")
        x = F.leaky_relu(self.fc1(x))
        print(f"    dqn x2: {x}")
        x = F.leaky_relu(self.fc2(x))
        print(f"    dqn x3: {x}")
        x = F.leaky_relu(self.fc3(x))
        print(f"    dqn x4: {x}")
        x = F.leaky_relu(self.fc4(x))
        print(f"    dqn x5: {x}")
        x = self.fc5(x)
        print(f"    dqn x6: {x}")
        return x