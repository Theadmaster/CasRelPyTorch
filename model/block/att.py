import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.query(x)  # Shape: (batch_size, seq_length, embed_size)
        K = self.key(x)  # Shape: (batch_size, seq_length, embed_size)
        V = self.value(x)  # Shape: (batch_size, seq_length, embed_size)

        # 计算注意力分数
        attention = torch.bmm(Q, K.transpose(1, 2)) / self.embed_size ** 0.5
        attention = F.softmax(attention, dim=-1)

        # 应用注意力权重
        out = torch.bmm(attention, V)
        return out
