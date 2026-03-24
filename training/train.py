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
import torch.multiprocessing as mp

from models.full_model import ColdStartModel
from preprocessing.attribute_builder import AttributeBuilder
from training.collate_fn import collate_fn
from loss.losses import total_loss
from loss.metrics import compute_metrics

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

#evaluate
def evaluate(model, dataloader, device):

    model.eval()

    metric_sums = {}
    count = 0

    with torch.no_grad():

        for batch in dataloader:

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

            scores, _, _ = model(
                exposure,
                click,
                semantic,
                histories,
                history_length_mask,
                candidates
            )

            # mask padded candidates
            scores = scores.masked_fill(candidate_mask == 0, -1e9)

            # -----------------------------
            # Compute metrics per batch
            # -----------------------------
            batch_metrics = compute_metrics(scores, labels)

            # accumulate
            for k, v in batch_metrics.items():
                if k not in metric_sums:
                    metric_sums[k] = 0.0
                metric_sums[k] += v

            count += 1

    # average over batches
    final_metrics = {k: v / count for k, v in metric_sums.items()}

    return final_metrics
# =========================================================
# Training
# =========================================================

def train(config):

    GLOBAL_BEST_PATH = os.path.join("best_model_global.pth")
    global_best_auc = 0.0
    patience = config["patience"]
    patience_counter = 0

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
        device="cpu",
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

    dev_behaviors_df = pd.read_csv(
    pv.DEV_BEHAVIORS_PATH,
    sep="\t",
    header=None,
    names=[
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ],)

    dev_dataset = MindDataset(
        dev_behaviors_df,
        attribute_builder,
        embedding_lookup
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn =collate_fn
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

    best_metrics = {"AUC": 0,"MRR": 0,"nDCG@5": 0}
    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(config["epochs"]):
        print(f"\nStarting Epoch {epoch+1}/{config['epochs']}")
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

            #gaurd for debugging
            if torch.isnan(loss):
                print("NaN detected!")
                print("Scores max:", scores.max())
                print("Scores min:", scores.min())
                print("Labels:", labels)
                break
            # -----------------------------
            # Backprop
            # -----------------------------

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        #calc metrics on dev set
        val_metrics = evaluate(model, dev_loader, device)

        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        current_auc = val_metrics["AUC"]

        # -------------------------
        # EARLY STOPPING LOGIC
        # -------------------------
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(" Early stopping triggered!")
            break
        # save best model
        if val_metrics["AUC"] > best_metrics["AUC"]:
            best_metrics = val_metrics

            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "best_model.pth")
            )

            print("Best model updated based on AUC")
        # -------------------------
        # GLOBAL BEST MODEL
        # -------------------------
        if val_metrics["AUC"] > global_best_auc:
            global_best_auc = val_metrics["AUC"]

            torch.save(model.state_dict(),GLOBAL_BEST_PATH)

            print("Global best model updated!")


# =========================================================
# Main
# =========================================================

def main():

    CONFIG = {

        "batch_size": 32,
        "epochs": 15,
        "lr": 0.0005,
        "lambda_align": 0.01,
        "embedding_dim": 384,
        "patience": 3,
        "num_workers": 0,
        "seed": 42

    }

    train(CONFIG)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
