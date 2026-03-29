import torch
import numpy as np


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def auc_score(scores, labels):
    """
    scores: (B, K)
    labels: (B,) -> index of clicked item
    """

    scores = to_numpy(scores)
    labels = to_numpy(labels)

    aucs = []

    for i in range(len(scores)):
        s = scores[i]
        y = np.zeros_like(s)
        y[labels[i]] = 1

        pos_score = s[labels[i]]
        neg_scores = np.delete(s, labels[i])

        # pairwise comparison
        correct = np.sum(pos_score > neg_scores)
        ties = np.sum(pos_score == neg_scores)

        auc = (correct + 0.5 * ties) / len(neg_scores)
        aucs.append(auc)

    return np.mean(aucs)


def mrr_score(scores, labels):
    scores = to_numpy(scores)
    labels = to_numpy(labels)

    mrr = []

    for i in range(len(scores)):
        ranking = np.argsort(-scores[i])  # descending
        rank = np.where(ranking == labels[i])[0][0] + 1
        mrr.append(1.0 / rank)

    return np.mean(mrr)


def ndcg_score(scores, labels, k=5):
    scores = to_numpy(scores)
    labels = to_numpy(labels)

    ndcgs = []

    for i in range(len(scores)):
        ranking = np.argsort(-scores[i])[:k]

        dcg = 0.0
        for j, idx in enumerate(ranking):
            if idx == labels[i]:
                dcg = 1.0 / np.log2(j + 2)
                break

        idcg = 1.0  # only one relevant item
        ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)



def compute_metrics(scores, labels, ks=[5, 10]):
    """
    Returns dictionary of all metrics
    """

    metrics = {
        "AUC": auc_score(scores, labels),
        "MRR": mrr_score(scores, labels),
    }

    for k in ks:
        metrics[f"nDCG@{k}"] = ndcg_score(scores, labels, k)

    return metrics