import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryEncoder(nn.Module):
    """
    History Encoder using self-attention pooling.

    Input:
        history_embeddings: (B, L, 384)

    Output:
        user_history_embedding: (B, 384)
    """

    def __init__(self, embedding_dim: int = 384):
        super(HistoryEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Step 1: Linear transformation W and bias b
        # Transforms each clicked embedding into importance space
        self.attention_linear = nn.Linear(embedding_dim, embedding_dim)

        # Step 2: Projection vector w
        # Converts transformed embedding into scalar importance score
        self.attention_score = nn.Linear(embedding_dim, 1)

    def forward(self, history_embeddings, history_mask=None, return_attention=False):

        """
        history_embeddings: (B, L, 384)
        history_mask: (B, L) → 1 for real history, 0 for padding
        """

        B, L, D = history_embeddings.shape

        if L == 0:
            return torch.zeros(B, D, device=history_embeddings.device)

        # Transform embeddings
        transformed = torch.tanh(self.attention_linear(history_embeddings))

        # Compute raw attention scores
        scores = self.attention_score(transformed).squeeze(-1)  # (B, L)

        # Apply mask BEFORE softmax
        if history_mask is not None:
            scores = scores.masked_fill(history_mask == 0, float("-inf"))

        # Softmax over valid positions only
        attention_weights = F.softmax(scores, dim=1)

        # Weighted sum
        user_history_embedding = torch.bmm(
            attention_weights.unsqueeze(1),
            history_embeddings
        ).squeeze(1)

        if return_attention:
            return user_history_embedding, attention_weights

        return user_history_embedding
    
if __name__ == "__main__":

    B = 1
    L = 4
    D = 384

    history_embeddings = torch.rand(B, L, D)

    encoder = HistoryEncoder()

    history_mask = torch.ones(B, L)
    output, weights = encoder(history_embeddings, history_mask, return_attention=True)

    print("Attention Weights:")
    print(weights)
    print("Sum of weights (should be 1):", weights.sum(dim=1))

    print("\nFinal User History Embedding:")
    print(output)