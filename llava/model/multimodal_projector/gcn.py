import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, hidden_size):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(4)])
        self.dropout = nn.Dropout(0.05)

    def laplace(self, A):
        device = A.device
        dtype = A.dtype
        # 计算行度矩阵 D_row
        D_row = A.sum(dim=-1)  # 每个源节点的度
        D_row_inv_sqrt = torch.pow(D_row, -0.5)  # 计算 D_row^(-1/2)
        D_row_inv_sqrt = torch.where(D_row == 0, torch.zeros_like(D_row, device=device, dtype=dtype), D_row_inv_sqrt)  # 避免度为0导致的inf

        # 计算列度矩阵 D_col
        D_col = A.sum(dim=-2)  # 每个目标节点的度
        D_col_inv_sqrt = torch.pow(D_col, -0.5)  # 计算 D_col^(-1/2)
        D_col_inv_sqrt = torch.where(D_col == 0, torch.zeros_like(D_col, device=device, dtype=dtype), D_col_inv_sqrt)  # 避免度为0导致的inf

        # 拉普拉斯归一化
        A_normalized = D_row_inv_sqrt.unsqueeze(-1) * A * D_col_inv_sqrt.unsqueeze(-2)

        return A_normalized

    def forward(self, node_embeddings, adj_matrix):
        adj_normalized = self.laplace(adj_matrix)
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        x = node_embeddings
        for layer in layers:
            # 图卷积计算
            x_next = layer(torch.bmm(adj_normalized, x))
            x_next = F.relu(x_next)
        x_next = x_next + x
        return self.dropout(x_next)
