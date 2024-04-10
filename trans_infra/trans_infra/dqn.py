import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """f(s) -> q : takes concat graph embeddings of states and predicts their q-vals"""
    def __init__(self, input_dim, hidden_dim, out_dim=1, dropout_p=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(p=dropout_p)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(p=dropout_p)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.drop3 = nn.Dropout(p=dropout_p)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim//2)
        self.norm4 = nn.BatchNorm1d(hidden_dim//2)
        self.drop4 = nn.Dropout(p=dropout_p)
        
        self.fc5 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.norm5 = nn.BatchNorm1d(hidden_dim//2)
        self.drop5 = nn.Dropout(p=dropout_p)
        
        self.fc6 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.norm6 = nn.BatchNorm1d(hidden_dim//2)
        self.drop6 = nn.Dropout(p=dropout_p)
        
        self.fc7 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.norm7 = nn.BatchNorm1d(hidden_dim//4)
        self.drop7 = nn.Dropout(p=dropout_p)
        
        self.fc8 = nn.Linear(hidden_dim//4, hidden_dim//4)
        self.norm8 = nn.BatchNorm1d(hidden_dim//4)
        self.drop8 = nn.Dropout(p=dropout_p)
        
        self.fc9 = nn.Linear(hidden_dim//4, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.drop1(self.norm1(self.fc1(x))))
        x = F.leaky_relu(self.drop2(self.norm2(self.fc2(x))))
        x = F.leaky_relu(self.drop3(self.norm3(self.fc3(x))))
        x = F.leaky_relu(self.drop4(self.norm4(self.fc4(x))))
        x = F.leaky_relu(self.drop5(self.norm5(self.fc5(x))))
        x = F.leaky_relu(self.drop6(self.norm6(self.fc6(x))))
        x = F.leaky_relu(self.drop7(self.norm7(self.fc7(x))))
        x = F.leaky_relu(self.drop8(self.norm8(self.fc8(x))))
        x = self.fc9(x)
        return x