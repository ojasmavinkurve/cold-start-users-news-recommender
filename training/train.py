import os
import pickle
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sympy import group
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


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingLookup:

    def __init__(self, embeddings, id_map):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)  
        self.id_map = id_map

    def __call__(self, nid):
        if nid in self.id_map:
            return self.embeddings[self.id_map[nid]]  
        return torch.zeros(384)



class MindDataset(Dataset):

    def __init__(self, behaviors_df, attribute_builder, embedding_lookup, num_negs, is_test):

        self.behaviors = behaviors_df
        self.attr_builder = attribute_builder
        self.embed = embedding_lookup
        self.cached_attrs = {}
        last_seen = {}
        self.is_test = is_test
        self.num_negs=num_negs

        for i in range(len(self.behaviors)):

            row = self.behaviors.iloc[i]

            curr_imp = row["impressions"]
            user_id = row["user_id"]

            if user_id not in last_seen:

                current_news_ids = [nid.split("-")[0] for nid in curr_imp.split()]

                attrs = {
                    "exposure": self.attr_builder.compute_exposure_vector(current_news_ids),
                    "click": torch.zeros(self.attr_builder.num_categories, device=self.attr_builder.device),
                    "semantic": torch.zeros(384, device=self.attr_builder.device)
                }

            else:
                prev_imp = last_seen[user_id]

                attrs = self.attr_builder.build_from_impression(prev_imp)

                current_news_ids = [nid.split("-")[0] for nid in curr_imp.split()]
                attrs["exposure"] = self.attr_builder.compute_exposure_vector(current_news_ids)

            self.cached_attrs[i] = attrs
            last_seen[user_id] = curr_imp
        
        print(len(self.cached_attrs), len(self.behaviors))

        if self.is_test:
            self.valid_indices = list(range(len(self.behaviors)))
        else:
            self.valid_indices = []
            for i, imp in enumerate(self.behaviors["impressions"]):
                if any(item.endswith("-1") for item in imp.split()):
                    self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.behaviors.iloc[real_idx]

        impressions = row["impressions"]
        history = row["history"]
        attrs = self.cached_attrs[real_idx]

        exposure = attrs["exposure"]
        click = attrs["click"]
        semantic = attrs["semantic"]

        #candidate embeddings and label with neg sampling
        if not self.is_test:
            NUM_NEGATIVES = self.num_negs  # tune this
            items = impressions.split()
            pos_items = []
            neg_items = []

                # separate positives and negatives
            for item in items:
                parts = item.split("-")
                nid = parts[0]

                if len(parts) == 2 and parts[1] == "1":
                    pos_items.append(nid)
                else:
                    neg_items.append(nid)

            # safety: skip if no positive
            if len(pos_items) == 0:
                return None

            # use only ONE positive (reduces noise)
            pos_nid = pos_items[0]

            # get category of positive
            pos_cat = self.attr_builder.news_to_category.get(pos_nid, None)

            hard_negs = []
            other_negs = []
            pos_emb = self.embed(pos_nid)

            neg_scores = []

            for nid in neg_items:
                neg_emb = self.embed(nid)

                sim = torch.cosine_similarity(
                    pos_emb.unsqueeze(0),
                    neg_emb.unsqueeze(0),
                    dim=-1
                ).item()

                neg_scores.append((nid, sim))

            # highest similarity = hardest negatives
            neg_scores.sort(key=lambda x: x[1], reverse=True)

            sampled_negs = [
                nid for nid, _ in neg_scores[:NUM_NEGATIVES]
            ]

            # split negatives into hard (same category) and others
            for nid in neg_items:
                cat = self.attr_builder.news_to_category.get(nid, None)

                if cat == pos_cat:
                    hard_negs.append(nid)
                else:
                    other_negs.append(nid)

            # sample negatives
            if len(hard_negs) >= NUM_NEGATIVES:
                sampled_negs = random.sample(hard_negs, NUM_NEGATIVES)
            else:
                remaining = NUM_NEGATIVES - len(hard_negs)
                sampled_negs = hard_negs + random.sample(
                    other_negs,
                    min(remaining, len(other_negs))
                )

            if self.is_test:
                final_items = pos_items + neg_items
            else:
                final_items = [pos_nid] + sampled_negs

            # shuffle to avoid position bias
            random.shuffle(final_items)

            # build embeddings and label
            candidates = []
            clicked_index = None

            for i, nid in enumerate(final_items):
                candidates.append(self.embed(nid))

                if nid == pos_nid:
                    clicked_index = i

            candidates = torch.stack(candidates)

            # IMPORTANT: keep label format SAME (list of indices)
            label = torch.tensor([clicked_index], dtype=torch.long)
            eval_label = torch.tensor(clicked_index, dtype=torch.long)

        else:
            candidates = []
            clicked_indices = []

            items = impressions.split()

            for i, item in enumerate(items):
                parts = item.split("-")
                nid = parts[0]

                candidates.append(self.embed(nid))

                if len(parts) == 2 and parts[1] == "1":
                    clicked_indices.append(i)

            candidates = torch.stack(candidates)

            if len(clicked_indices) > 0:
                label = torch.tensor(clicked_indices, dtype=torch.long)
                eval_label = torch.tensor(clicked_indices[0], dtype=torch.long)
            else:
                label = torch.tensor([], dtype=torch.long)
                eval_label = torch.tensor(-1, dtype=torch.long)
       
        #history embeddings
        history_embeddings = []

        if isinstance(history, str):
            for nid in history.split():
                history_embeddings.append(self.embed(nid))
                
        if len(history_embeddings) > 0:
            history_embeddings = torch.stack(history_embeddings)
            history_mask = torch.tensor(1.0)

        else:
            history_embeddings = torch.zeros(0, 384)
            history_mask = torch.tensor(0.0)

        return (
            exposure,
            click,
            semantic,
            history_embeddings,
            candidates,
            label,
            eval_label,
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
                eval_labels,
                history_masks,
                history_length_mask,
                candidate_mask
            ) = batch

            exposure = exposure.to(device)
            click = click.to(device)
            semantic = semantic.to(device)

            histories = histories.to(device)
            candidates = candidates.to(device)
            #labels = [l.to(device) for l in labels]

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
            #scores = scores + (candidate_mask + 1e-45).log()

            scores = scores.cpu()
            labels=eval_labels.to(device)

            labels = labels.cpu()

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

def train(config):

    GLOBAL_BEST_PATH = os.path.join("best_model_global.pth")
    global_best_mrr = 0.0
    best_mrr=0.0
    patience = config["patience"]
    patience_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    set_seed(config["seed"])


    run_dir = os.path.join(
        "runs",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    os.makedirs(run_dir, exist_ok=True)

    print("Run directory:", run_dir)


    with open(pv.CATEGORY_INDEX_PATH, "rb") as f:
        category_index = pickle.load(f)

    with open(pv.NEWS_ID_TO_INDEX_PATH, "rb") as f:
        news_id_to_index = pickle.load(f)

    news_embeddings = np.load(pv.NEWS_EMBEDDINGS_PATH)

    embedding_lookup = EmbeddingLookup(
        news_embeddings,
        news_id_to_index
    )


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
        embedding_lookup,
        num_negs = config["num_negs"],
        is_test=False
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
        embedding_lookup, 
        num_negs = config["num_negs"],
        is_test=False
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn =collate_fn
    )


    model = ColdStartModel(
        num_categories=len(category_index),
        embedding_dim=config["embedding_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]   
    )
    best_loss = float("inf")

    best_metrics = {"AUC": 0,"MRR": 0,"nDCG@5": 0}

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
                eval_labels,
                history_masks,
                history_length_mask,
                candidate_mask
            ) = batch

            exposure = exposure.to(device)
            click = click.to(device)
            semantic = semantic.to(device)

            histories = histories.to(device)
            candidates = candidates.to(device)

            labels =[l.to(device) for l in labels]
            
            history_length_mask = history_length_mask.to(device)
            candidate_mask = candidate_mask.to(device)

            #forward
            scores, u_attr, u_hist = model(
                exposure,
                click,
                semantic,
                histories,
                history_length_mask,
                candidates
            )

            #mask padded candidates
            scores = scores.masked_fill(
                candidate_mask == 0,
                -1e9
            )

            #filtering invalid labels for list of tensors
            valid_indices = [i for i, l in enumerate(labels) if l.numel() > 0]

            if len(labels) == 0:
                continue

            scores = scores[valid_indices]
            labels = [labels[i] for i in valid_indices]
            u_attr = u_attr[valid_indices]
            u_hist = u_hist[valid_indices]
            history_mask = history_masks.squeeze().to(device)[valid_indices]

            loss, rec_loss, align_loss = total_loss(
                scores,
                labels,
                u_attr,
                u_hist,
                history_mask,
                config["lambda_align"]
            )

            #l2 reg or weight decay to prevent overfitting
            #l2_reg = 0.0
            #lambda_reg = config["lambda_reg"]
            #for param in model.parameters():
                #l2_reg += torch.norm(param, 2) #iterated and calculated euclidean norm of every weight and bias

            #loss = loss + lambda_reg * l2_reg
            
            #gaurd for debugging
            if torch.isnan(loss):
                print("NaN detected!")
                print("Scores max:", scores.max())
                print("Scores min:", scores.min())
                print("Labels:", labels)
                break
            
            #backpropagation
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
        
        current_mrr = val_metrics["MRR"]

        #early stopping
        if current_mrr > best_mrr:
            best_mrr= current_mrr
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(" Early stopping triggered!")
            break
        # save best model
        if val_metrics["MRR"] > best_metrics["MRR"]:
            best_metrics = val_metrics

            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "best_model.pth")
            )

            print("Best model updated based on MRR")
        #global best model save
        if val_metrics["MRR"] > global_best_mrr:
            global_best_mrr = val_metrics["MRR"]

            torch.save(model.state_dict(),GLOBAL_BEST_PATH)

            print("Global best model updated!")



def main():

    CONFIG = {

        "batch_size": 32,
        "epochs": 15,
        "num_negs": 20,
        "lr": 0.0005,
        "lambda_align": 0.07,
        "weight_decay":0.00005,
        "embedding_dim": 384,
        "patience": 3,
        "num_workers": 0,
        "seed": 42

    }

    train(CONFIG)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
