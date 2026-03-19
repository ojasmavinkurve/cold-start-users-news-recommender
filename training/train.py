import os
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.full_model import ColdStartModel
from preprocessing.attribute_builder import AttributeBuilder
from training.collate_fn import collate_fn
from loss.losses import total_loss

import path_variables as pv


# =========================================================
# Utilities
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Embedding Lookup
# =========================================================

class EmbeddingLookup:

    def __init__(self, embeddings, id_map):

        self.embeddings = embeddings
        self.id_map = id_map

    def __call__(self, nid):

        if nid in self.id_map:
            idx = self.id_map[nid]
            return torch.tensor(self.embeddings[idx], dtype=torch.float32)

        return torch.zeros(384)


# =========================================================
# Dataset
# =========================================================

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

        # -----------------------------
        # Attribute vectors
        # -----------------------------

        attrs = self.attr_builder.build_from_impression(impressions)

        exposure = attrs["exposure"]
        click = attrs["click"]
        semantic = attrs["semantic"]

        # -----------------------------
        # Candidate embeddings
        # -----------------------------

        candidates = []
        clicked_index = None

        for i, item in enumerate(impressions.split()):

            nid, label = item.split("-")

            candidates.append(self.embed(nid))

            if label == "1":
                clicked_index = i

        # 🚨 Handle no-click case
        if clicked_index is None:
            return self.__getitem__((idx + 1) % len(self))

        candidates = torch.stack(candidates)

        label = torch.tensor(clicked_index, dtype=torch.long)

        # -----------------------------
        # History embeddings
        # -----------------------------

        history_embeddings = []

        if isinstance(history, str):

            for nid in history.split():
                history_embeddings.append(self.embed(nid))

        if len(history_embeddings) > 0:

            history_embeddings = torch.stack(history_embeddings)
            history_mask = torch.tensor([1.0])

        else:

            history_embeddings = torch.zeros(0, 384)
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


# =========================================================
# Training
# =========================================================

def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    set_seed(config["seed"])

    # -----------------------------------------------------
    # Run folder
    # -----------------------------------------------------

    run_dir = os.path.join(
        "runs",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    os.makedirs(run_dir, exist_ok=True)

    print("Run directory:", run_dir)

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
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
    )

    dev_news = pd.read_csv(
        pv.DEV_NEWS_PATH,
        sep="\t",
        header=None,
        names=train_news.columns,
    )

    news_df = pd.concat([train_news, dev_news])

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
        device=device,
        verbose=False
    )

    # -----------------------------------------------------
    # Load behaviors
    # -----------------------------------------------------

    behaviors_df = pd.read_csv(
        pv.TRAIN_BEHAVIORS_PATH,
        sep="\t",
        header=None,
        names=[
            "impression_id",
            "user_id",
            "time",
            "history",
            "impressions",
        ],
    )

    dataset = MindDataset(
        behaviors_df,
        attribute_builder,
        embedding_lookup
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = ColdStartModel(
        num_categories=len(category_index),
        embedding_dim=config["embedding_dim"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    best_loss = float("inf")

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(config["epochs"]):

        model.train()

        total_loss_epoch = 0

        progress = tqdm(dataloader)

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

            # -----------------------------
            # Forward
            # -----------------------------

            scores, u_attr, u_hist = model(
                exposure,
                click,
                semantic,
                histories,
                history_length_mask,
                candidates
            )

            # -----------------------------
            # Mask padded candidates
            # -----------------------------

            scores = scores.masked_fill(
                candidate_mask == 0,
                -1e9
            )

            history_mask = history_masks.squeeze().to(device)

            loss, rec_loss, align_loss = total_loss(
                scores,
                labels,
                u_attr,
                u_hist,
                history_mask,
                config["lambda_align"]
            )

            # -----------------------------
            # Backprop
            # -----------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss_epoch += loss.item()

            progress.set_description(
                f"Epoch {epoch+1} | Loss {loss.item():.4f}"
            )

        avg_loss = total_loss_epoch / len(dataloader)

        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # save epoch model

        torch.save(
            model.state_dict(),
            os.path.join(run_dir, f"epoch_{epoch+1}.pth")
        )

        # save best model

        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "best_model.pth")
            )

            print("Best model updated")


# =========================================================
# Main
# =========================================================

def main():

    CONFIG = {

        "batch_size": 32,
        "epochs": 5,
        "lr": 1e-4,
        "lambda_align": 0.01,
        "embedding_dim": 384,
        "num_workers": 0,
        "seed": 42

    }

    train(CONFIG)


if __name__ == "__main__":

    main()