import torch
import torch.nn as nn


class MultiHeadCausalAttention(nn.Module):
    """SCM message passing: h_i <- sum_j A_ij W h_j.

    The name is kept for compatibility with the rest of the prototype, but
    this is deliberately not QKV transformer attention.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.message = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        messages = torch.matmul(adjacency, hidden)
        messages = self.message(messages)
        messages = self.dropout(messages)
        return self.out(messages), adjacency


class CausalAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = MultiHeadCausalAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if adjacency is None:
            # ablation "no causal graph": objects exchange no messages, each node is
            # only refined by its own feed-forward network
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
            return hidden, None
        propagated, graph = self.attention(self.norm1(hidden), adjacency)
        hidden = hidden + self.dropout(propagated)
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        return hidden, graph
