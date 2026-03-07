import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.full_model import ColdStartModel
from preprocessing.attribute_builder import AttributeBuilder
from training.collate_fn import collate_fn
from loss.losses import total_loss

import path_variables as pv


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class MindDataset(Dataset):

    def __init__(self, behaviors_df, attribute_builder, embedding_lookup):

        self.behaviors = behaviors_df
        self.attr_builder = attribute_builder
        self.embed = embedding_lookup

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):

        row = self.behaviors.iloc[idx]

        impressions = row["impressions"]
        history = row["history"]

        # reuse attribute builder
        attrs = self.attr_builder.build_from_impression(impressions)

        exposure = attrs["exposure"]
        click = attrs["click"]
        semantic = attrs["semantic"]

        # candidates
        candidate_ids = []
        clicked_index = None

        candidates = []

        for i, item in enumerate(impressions.split()):

            nid, label = item.split("-")

            candidates.append(self.embed(nid))

            if label == "1":
                clicked_index = i

        candidates = torch.stack(candidates)

        label = torch.tensor(clicked_index)

        # history embeddings
        history_embeddings = []

        if isinstance(history, str):

            for nid in history.split():
                history_embeddings.append(self.embed(nid))

        if len(history_embeddings) > 0:
            history_embeddings = torch.stack(history_embeddings)
            history_mask = torch.tensor([1.0])
        else:
            history_embeddings = torch.zeros(0,384)
            history_mask = torch.tensor([0.0])

        return (
            exposure,
            click,
            semantic,
            history_embeddings,
            candidates,
            label,
            history_mask
        )

# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # -----------------------------------------------------
    # Load preprocessed files
    # -----------------------------------------------------

    print("Loading preprocessing artifacts...")

    with open(pv.CATEGORY_INDEX_PATH, "rb") as f:
        category_index = pickle.load(f)

    with open(pv.NEWS_ID_TO_INDEX_PATH, "rb") as f:
        news_id_to_index = pickle.load(f)

    news_embeddings = np.load(pv.NEWS_EMBEDDINGS_PATH)

    # convert embeddings dict for attribute builder
    news_embeddings_dict = {
        nid: torch.tensor(news_embeddings[idx], dtype=torch.float32)
        for nid, idx in news_id_to_index.items()
    }

    # -----------------------------------------------------
    # Load news dataframe
    # -----------------------------------------------------

    train_news = pd.read_csv(
        pv.TRAIN_NEWS_PATH,
        sep="\t",
        header=None,
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"
        ],
    )

    dev_news = pd.read_csv(
        pv.DEV_NEWS_PATH,
        sep="\t",
        header=None,
        names=[
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"
        ],
    )

    news_df = pd.concat([train_news, dev_news])

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = MindDataset(
        behaviors_path=pv.TRAIN_BEHAVIORS_PATH,
        news_df=news_df,
        category_index=category_index,
        news_embeddings=news_embeddings,
        news_id_to_index=news_id_to_index,
        device=device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = ColdStartModel(
        num_categories=len(category_index),
        embedding_dim=384
    )

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 5
    lambda_align = 0.01

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_epoch_loss = 0

        progress = tqdm(train_loader)

        for batch in progress:

            (
                exposure,
                click,
                semantic,
                histories,
                candidates,
                labels,
                history_masks,
                history_length_mask,
                candidate_mask
            ) = batch

            exposure = exposure.to(device)
            click = click.to(device)
            semantic = semantic.to(device)
            histories = histories.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)
            history_length_mask = history_length_mask.to(device)
            candidate_mask = candidate_mask.to(device)

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            scores, u_attr, u_hist = model(
                exposure,
                click,
                semantic,
                histories,
                history_length_mask,
                candidates
            )

            # -------------------------------------------------
            # Mask padded candidates
            # -------------------------------------------------

            scores = scores.masked_fill(
                candidate_mask == 0,
                float("-inf")
            )

            # history mask for alignment
            history_mask = history_masks.squeeze().to(device)

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------

            loss, rec_loss, align_loss = total_loss(
                scores,
                labels,
                u_attr,
                u_hist,
                history_mask,
                lambda_align
            )

            # -------------------------------------------------
            # Backprop
            # -------------------------------------------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            progress.set_description(
                f"Epoch {epoch+1} | Loss {loss.item():.4f}"
            )

        avg_loss = total_epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} finished | Avg Loss: {avg_loss:.4f}")

        # save checkpoint
        torch.save(
            model.state_dict(),
            f"model_epoch_{epoch+1}.pth"
        )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    train()