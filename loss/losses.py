import torch
import torch.nn.functional as F


def recommendation_loss(scores, labels):
    """
    Cross-entropy recommendation loss.

    Args:
        scores: (B, K) raw prediction scores for candidate news
        labels: (B,) index of clicked news in candidate list

    Returns:
        Scalar cross-entropy loss
    """
    return F.cross_entropy(scores, labels)


def alignment_loss(u_attr, u_hist, history_mask):
    """
    Attribute–Behavior alignment loss.

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

    # Compute per-user MSE (without reduction)
    mse_per_user = F.mse_loss(u_attr, u_hist, reduction='none')  # (B, 384)

    # Sum over embedding dimensions
    mse_per_user = mse_per_user.mean(dim=1)  # (B,)

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
               lambda_align=0.01):
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
    rec_loss = recommendation_loss(scores, labels)

    # Alignment loss (only for users with history)
    align_loss = alignment_loss(u_attr, u_hist, history_mask)

    # Total loss
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

    rec = recommendation_loss(scores, labels)
    align = alignment_loss(u_attr, u_hist, history_mask)
    total, rec_l, align_l = total_loss(
        scores, labels, u_attr, u_hist, history_mask, lambda_align=0.01
    )

    print("Recommendation Loss:", rec.item())
    print("Alignment Loss:", align.item())
    print("Total Loss:", total.item())