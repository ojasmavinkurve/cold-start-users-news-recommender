import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module for combining attribute and history embeddings.

    Input:
        u_attr: (B, 384)
        u_hist: (B, 384)

    Output:
        final_user_embedding: (B, 384)
    """

    def __init__(self, embedding_dim: int = 384):
        super(AdaptiveFusion, self).__init__()

        self.embedding_dim = embedding_dim

        # Fusion gate layer
        # Input size = 384 + 384 = 768
        # Output size = 384 (dimension-wise gating)
        self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, u_attr, u_hist):
        """
        u_attr: (B, 384)
        u_hist: (B, 384)
        """

        # Step 1: Concatenate
        concat = torch.cat([u_attr, u_hist], dim=-1)  # (B, 768)

        # Step 2: Compute gate values
        gate = torch.sigmoid(self.fusion_layer(concat))  # (B, 384)

        # Step 3: Dimension-wise interpolation
        final_user_embedding = (
            gate * u_hist +
            (1 - gate) * u_attr
        )

        return final_user_embedding
    
if __name__ == "__main__":

    B = 2
    D = 384

    u_attr = torch.rand(B, D)
    u_hist = torch.rand(B, D)

    fusion = AdaptiveFusion()

    output = fusion(u_attr, u_hist)

    print("Attribute shape:", u_attr.shape)
    print("History shape:", u_hist.shape)
    print("Final user embedding shape:", output.shape)