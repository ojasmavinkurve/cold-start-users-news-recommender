import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.full_model import ColdStartModel
from preprocessing.attribute_builder import AttributeBuilder
from training.collate_fn import collate_fn
from training.train import EmbeddingLookup, MindDataset, evaluate  # reuse directly
from loss.metrics import compute_metrics  

import path_variables as pv


# =========================================================
# Evaluation
# =========================================================

# def evaluate(model, dataloader, device):

#     model.eval()
#     metric_sums = {}
#     count = 0

#     all_scores = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader):

#             (
#                 exposure,
#                 click,
#                 semantic,
#                 histories,
#                 candidates,
#                 labels,
#                 history_masks,
#                 history_length_mask,
#                 candidate_mask
#             ) = batch

#             exposure = exposure.to(device)
#             click = click.to(device)
#             semantic = semantic.to(device)

#             histories = histories.to(device)
#             candidates = candidates.to(device)

#             labels = labels.to(device)
#             candidate_mask = candidate_mask.to(device)
#             history_length_mask = history_length_mask.to(device)

#             # -------------------------
#             # Forward pass
#             # -------------------------
#             scores, _, _ = model(
#                 exposure,
#                 click,
#                 semantic,
#                 histories,
#                 history_length_mask,
#                 candidates
#             )

#             # mask padding
#             scores = scores.masked_fill(candidate_mask == 0, -1e9)
#             batch_metrics = compute_metrics(scores, labels)

#             for k, v in batch_metrics.items():
#                 metric_sums[k] = metric_sums.get(k, 0) + v

#             count += 1


#             all_scores.append(scores)
#             all_labels.append(labels)

#     all_scores = torch.cat(all_scores, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)

#     final_metrics = {k: v / count for k, v in metric_sums.items()}

#     return final_metrics


# =========================================================
# Main
# =========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------------------------------------
    # Load preprocessing artifacts
    # -----------------------------------------------------

    with open(pv.CATEGORY_INDEX_PATH, "rb") as f:
        category_index = pickle.load(f)

    with open(pv.NEWS_ID_TO_INDEX_PATH, "rb") as f:
        news_id_to_index = pickle.load(f)

    news_embeddings = np.load(pv.NEWS_EMBEDDINGS_PATH)

    embedding_lookup = EmbeddingLookup(
        news_embeddings,
        news_id_to_index
    )

    # -----------------------------------------------------
    # Load news
    # -----------------------------------------------------

    train_news = pd.read_csv(
        pv.TRAIN_NEWS_PATH,
        sep="\t",
        header=None,
        names=[
            "news_id", "category", "subcategory",
            "title", "abstract", "url",
            "title_entities", "abstract_entities"
        ]
    )

    dev_news = pd.read_csv(
        pv.DEV_NEWS_PATH,
        sep="\t",
        header=None,
        names=train_news.columns
    )

    test_news = pd.read_csv(
        pv.TEST_NEWS_PATH,
        sep="\t",
        header=None,
        names=train_news.columns
    )


    news_df = pd.concat([train_news, dev_news, test_news])
    news_df = news_df.drop_duplicates(subset=["news_id"])

    # -----------------------------------------------------
    # Attribute Builder
    # -----------------------------------------------------

    news_embeddings_dict = {
        nid: torch.tensor(news_embeddings[idx], dtype=torch.float32)
        for nid, idx in news_id_to_index.items()
    }

    attribute_builder = AttributeBuilder(
        news_df=news_df,
        category_index=category_index,
        news_embeddings=news_embeddings_dict,
        verbose=False,
        is_test=True
    )

    # -----------------------------------------------------
    # Load TEST (DEV) behaviors
    # -----------------------------------------------------

    behaviors_df = pd.read_csv(
        pv.TEST_BEHAVIORS_PATH,
        sep="\t",
        header=None,
        names=[
            "impression_id",
            "user_id",
            "time",
            "history",
            "impressions"
        ]
    )

    dataset = MindDataset(
        behaviors_df,
        attribute_builder,
        embedding_lookup
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = ColdStartModel(
        num_categories=len(category_index)
    ).to(device)

    model.load_state_dict(torch.load("best_model_global.pth", map_location=device, weights_only=True))
    
    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    metrics = evaluate(model, dataloader, device)

    print("\n===== TEST METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()