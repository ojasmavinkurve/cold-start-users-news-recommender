import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeEncoder(nn.Module):
    """
    Attribute Encoder using gated projection fusion.

    Input:
        exposure: (B, K)
        click:    (B, K)
        semantic: (B, 384)

    Output:
        user_attr_embedding: (B, 384)
    """

    def __init__(self, num_categories: int, embedding_dim: int = 384):
        super(AttributeEncoder, self).__init__()

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        # Projection layers
        # Exposure: K -> 384
        self.exposure_proj = nn.Linear(num_categories, embedding_dim)

        # Click: K -> 384
        self.click_proj = nn.Linear(num_categories, embedding_dim)

        # Semantic: 384 -> 384
        self.semantic_proj = nn.Linear(embedding_dim, embedding_dim)

        # Gating layer
        # Input size = exposure(K) + click(K) + semantic(384)
        gate_input_dim = num_categories * 2 + embedding_dim
        self.gate_layer = nn.Linear(gate_input_dim, 3)

    def forward(self, exposure, click, semantic):
        """
        exposure: (B, K)
        click:    (B, K)
        semantic: (B, 384)
        """

        # Project each attribute to 384-dimensional space
        e_proj = self.exposure_proj(exposure)   # (B, 384)
        c_proj = self.click_proj(click)         # (B, 384)
        s_proj = self.semantic_proj(semantic)   # (B, 384)

        # Compute gating weights
        raw_concat = torch.cat([exposure, click, semantic], dim=-1)
        gate_scores = self.gate_layer(raw_concat)  # (B, 3)

        # Convert scores to probabilities
        gates = F.softmax(gate_scores, dim=-1)     # (B, 3)

        # Split gate weights
        g_exp = gates[:, 0].unsqueeze(-1)  # (B, 1)
        g_clk = gates[:, 1].unsqueeze(-1)  # (B, 1)
        g_sem = gates[:, 2].unsqueeze(-1)  # (B, 1)

        # Weighted fusion
        user_attr_embedding = (
            g_exp * e_proj +
            g_clk * c_proj +
            g_sem * s_proj
        )

        return user_attr_embedding

if __name__ == "__main__":
    B = 2
    K = 17

    encoder = AttributeEncoder(num_categories=K)

    exposure = torch.rand(B, K)
    click = torch.rand(B, K)
    semantic = torch.rand(B, 384)

    output = encoder(exposure, click, semantic)

    print("Output shape:", output.shape)