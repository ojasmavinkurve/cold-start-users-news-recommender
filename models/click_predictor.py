import torch
import torch.nn as nn


class ClickPredictor(nn.Module):
    """
    Click prediction layer.

    Computes dot-product relevance score between
    final user embedding and candidate news embeddings.

    Input:
        user_embedding: (B, 384)
        candidate_embeddings: (B, K, 384)

    Output:
        scores: (B, K)
    """

    def __init__(self):
        super(ClickPredictor, self).__init__()

    def forward(self, user_embedding, candidate_embeddings):
        """
        user_embedding: (B, 384)
        candidate_embeddings: (B, K, 384)
        """

        # Expand user embedding for batch matrix multiplication
        # (B, 384) -> (B, 384, 1)
        user_embedding = user_embedding.unsqueeze(-1)

        # Batch matrix multiply:
        # (B, K, 384) x (B, 384, 1) -> (B, K, 1)
        scores = torch.bmm(candidate_embeddings, user_embedding)

        # Remove last dimension
        scores = scores.squeeze(-1)  # (B, K)

        return scores
    
if __name__ == "__main__":

    B = 2
    K = 5
    D = 384

    user_embedding = torch.randn(B, D)
    candidate_embeddings = torch.randn(B, K, D)

    predictor = ClickPredictor()

    scores = predictor(user_embedding, candidate_embeddings)

    print("User embedding shape:", user_embedding.shape)
    print("Candidate embeddings shape:", candidate_embeddings.shape)
    print("Scores shape:", scores.shape)
    print("Scores:", scores)