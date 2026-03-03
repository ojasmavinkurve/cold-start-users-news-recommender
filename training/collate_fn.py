import torch


def collate_fn(batch):
    """
    Custom collate function for MindDataset.
    Handles variable history length and candidate length.
    """

    (
        exposures,
        clicks,
        semantics,
        histories,
        candidates,
        labels,
        history_masks
    ) = zip(*batch)

    # -----------------------------------------
    # Stack simple tensors
    # -----------------------------------------
    exposures = torch.stack(exposures)
    clicks = torch.stack(clicks)
    semantics = torch.stack(semantics)
    labels = torch.stack(labels)
    history_masks = torch.stack(history_masks)

    # -----------------------------------------
    # Pad history embeddings
    # -----------------------------------------
    max_history_len = max(h.shape[0] for h in histories)

    padded_histories = []
    history_length_mask = []

    for h in histories:
        L, D = h.shape
        pad_len = max_history_len - L

        if pad_len > 0:
            padding = torch.zeros(pad_len, D, device=h.device)
            h = torch.cat([h, padding], dim=0)

        padded_histories.append(h)

        mask = torch.zeros(max_history_len, device=h.device)
        mask[:L] = 1
        history_length_mask.append(mask)

    padded_histories = torch.stack(padded_histories)
    history_length_mask = torch.stack(history_length_mask)

    # -----------------------------------------
    # Pad candidate embeddings
    # -----------------------------------------
    max_candidate_len = max(c.shape[0] for c in candidates)

    padded_candidates = []
    candidate_mask = []

    for c in candidates:
        C, D = c.shape
        pad_len = max_candidate_len - C

        if pad_len > 0:
            padding = torch.zeros(pad_len, D, device=c.device)
            c = torch.cat([c, padding], dim=0)

        padded_candidates.append(c)

        mask = torch.zeros(max_candidate_len, device=c.device)
        mask[:C] = 1
        candidate_mask.append(mask)

    padded_candidates = torch.stack(padded_candidates)
    candidate_mask = torch.stack(candidate_mask)

    return (
        exposures,
        clicks,
        semantics,
        padded_histories,
        padded_candidates,
        labels,
        history_masks,
        history_length_mask,
        candidate_mask
    )