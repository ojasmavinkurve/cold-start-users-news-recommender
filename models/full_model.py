import torch
import torch.nn as nn

from .attribute_encoder import AttributeEncoder
from .history_encoder import HistoryEncoder
from .adaptive_fusion import AdaptiveFusion
from .click_predictor import ClickPredictor


class ColdStartModel(nn.Module):
    """
    Full Cold-Start News Recommendation Model
    """

    def __init__(
        self,
        num_categories: int,
        embedding_dim: int = 384,
        use_history: bool = True,
        use_fusion: bool = True
    ):
        super(ColdStartModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_history = use_history
        self.use_fusion = use_fusion

        self.attribute_encoder = AttributeEncoder(
            num_categories=num_categories,
            embedding_dim=embedding_dim
        )

        self.history_encoder = HistoryEncoder(
            embedding_dim=embedding_dim
        )

        self.fusion = AdaptiveFusion(
            embedding_dim=embedding_dim
        )

        self.click_predictor = ClickPredictor()

    # ✅ FORWARD MUST BE INSIDE CLASS
    def forward(
        self,
        exposure,
        click,
        semantic,
        history_embeddings,
        history_length_mask,
        candidate_embeddings
    ):
        """
        exposure: (B, K)
        click: (B, K)
        semantic: (B, 384)
        history_embeddings: (B, L, 384)
        history_length_mask: (B, L)
        candidate_embeddings: (B, C, 384)
        """

        # Attribute Representation
        u_attr = self.attribute_encoder(
            exposure,
            click,
            semantic
        )

        # History Representation (mask-aware)
        if self.use_history and history_embeddings is not None:
            u_hist = self.history_encoder(
                history_embeddings,
                history_mask=history_length_mask
            )
        else:
            B = exposure.size(0)
            device = exposure.device
            u_hist = torch.zeros(B, self.embedding_dim, device=device)

        # Adaptive Fusion
        if self.use_fusion:
            user_embedding = self.fusion(u_attr, u_hist)
        else:
            user_embedding = u_attr

        # Click Prediction
        scores = self.click_predictor(
            user_embedding,
            candidate_embeddings
        )

        return scores, u_attr, u_hist


# TEST BLOCK
if __name__ == "__main__":

    B = 2
    K = 18
    L = 5
    C = 10
    D = 384

    model = ColdStartModel(num_categories=K)

    exposure = torch.rand(B, K)
    click = torch.rand(B, K)
    semantic = torch.rand(B, D)
    history_embeddings = torch.rand(B, L, D)
    history_length_mask = torch.ones(B, L)
    candidate_embeddings = torch.rand(B, C, D)

    scores, u_attr, u_hist = model(
        exposure,
        click,
        semantic,
        history_embeddings,
        history_length_mask,
        candidate_embeddings
    )

    print("Scores shape:", scores.shape)
    print("Attribute embedding shape:", u_attr.shape)
    print("History embedding shape:", u_hist.shape)