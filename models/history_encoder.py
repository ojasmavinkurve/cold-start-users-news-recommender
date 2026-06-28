import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryEncoder(nn.Module):

    def __init__(self, embedding_dim=384):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.attn_pool = nn.Linear(embedding_dim, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        history_embeddings,
        history_mask=None,
        return_attention=False
    ):

        B, L, D = history_embeddings.shape

        if L == 0:
            return torch.zeros(
                B,
                D,
                device=history_embeddings.device
            )

        # self-attention
        if history_mask is not None:

            key_padding_mask = (history_mask == 0)

            # avoid all-masked rows
            all_masked = key_padding_mask.all(dim=1)

            if all_masked.any():
                key_padding_mask[all_masked, 0] = False

        else:
            key_padding_mask = None


        attn_output, _ = self.self_attn(
            history_embeddings,
            history_embeddings,
            history_embeddings,
            key_padding_mask=key_padding_mask
        )

        attn_output = torch.nan_to_num(attn_output)
        attn_output = self.dropout(attn_output)
       
        # pooling scores
        scores = self.attn_pool(attn_output).squeeze(-1)

        # mask padding
        if history_mask is not None:
            scores = scores.masked_fill(
                history_mask == 0,
                -1e9
            )

        # attention weights
        attention_weights = F.softmax(scores, dim=1)

        # weighted sum
        user_history_embedding = torch.bmm(
            attention_weights.unsqueeze(1),
            attn_output
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