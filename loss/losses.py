import torch
import torch.nn.functional as F


# def recommendation_loss(scores, labels):
#     """
#     Cross-entropy recommendation loss.

#     Args:
#         scores: (B, K) raw prediction scores for candidate news
#         labels: (B,) index of clicked news in candidate list

#     Returns:
#         Scalar cross-entropy loss
#     """
#     return F.cross_entropy(scores, labels)

def bpr_loss(scores, labels):

    total_loss = 0.0
    valid_count = 0

    B, K = scores.shape

    for i in range(B):

        pos_indices = labels[i]

        if isinstance(pos_indices, list):
            pos_indices = torch.tensor(pos_indices, device=scores.device)

        if pos_indices.numel() == 0:
            continue

        pos_indices = pos_indices.view(-1)

        pos_scores = scores[i][pos_indices]

        neg_mask = torch.ones(K, dtype=torch.bool, device=scores.device)
        neg_mask[pos_indices] = False

        neg_scores = scores[i][neg_mask]

        #skip if no negatives or invalid shape
        if neg_scores.numel() == 0:
            continue

        # ensure proper shape
        pos_scores = pos_scores.view(-1, 1)   # (P,1)
        neg_scores = neg_scores.view(1, -1)   # (1,N)

        diff = pos_scores - neg_scores        # (P,N)

        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        total_loss += loss
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=scores.device)

    return total_loss / valid_count


def alignment_loss(u_attr, u_hist, history_mask):
    """
    Attribute Behavior alignment loss.

    Computes MSE between attribute embedding and history embedding,
    but only for users who have history.

    Args:
        u_attr: (B, 384)
        u_hist: (B, 384)
        history_mask: (B,) tensor with:
                      1 -> user has history
                      0 -> zero-history user

    Returns:
        Scalar alignment loss
    """

    cos_sim = F.cosine_similarity(u_attr, u_hist, dim=-1)

    mse_per_user = 1 - cos_sim

    history_mask = history_mask.float()
    # Apply mask (only users with history)
    masked_mse = mse_per_user * history_mask

    # Avoid division by zero
    if history_mask.sum() > 0:
        return masked_mse.sum() / history_mask.sum()
    else:
        return torch.tensor(0.0, device=u_attr.device)


def total_loss(scores,
               labels,
               u_attr,
               u_hist,
               history_mask,
               lambda_align): #reduced lambda_align to prevent overfitting
    """
    Total training loss.

    L_total = L_rec + lambda * L_align

    Args:
        scores: (B, K)
        labels: (B,)
        u_attr: (B, 384)
        u_hist: (B, 384)
        history_mask: (B,)
        lambda_align: scalar hyperparameter

    Returns:
        Scalar total loss
    """

    # Recommendation loss
    rec_loss = bpr_loss(scores, labels)

    # Alignment loss (only for users with history)
    #when users have less history, the alignment loss can be noisy. we donot consider users with less history 
    align_loss = alignment_loss(u_attr, u_hist, history_mask)
 
    loss = rec_loss + lambda_align * align_loss


    return loss, rec_loss, align_loss

if __name__ == "__main__":

    print("Testing losses module...\n")

    B = 3
    K = 5
    D = 384

    scores = torch.randn(B, K)
    labels = torch.tensor([1, 3, 0])

    u_attr = torch.randn(B, D)
    u_hist = torch.randn(B, D)

    history_mask = torch.tensor([1.0, 0.0, 1.0])

    rec = bpr_loss(scores, labels)
    align = alignment_loss(u_attr, u_hist, history_mask)
    total, rec_l, align_l = total_loss(
        scores, labels, u_attr, u_hist, history_mask, lambda_align=0.01
    )

    print("Recommendation Loss:", rec.item())
    print("Alignment Loss:", align.item())
    print("Total Loss:", total.item())
