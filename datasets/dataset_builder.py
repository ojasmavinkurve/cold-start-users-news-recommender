import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle

import path_variables as pv
from preprocessing.attribute_builder import AttributeBuilder


class MindDataset(Dataset):
    """
    MIND Dataset for Cold-Start News Recommendation.

    Each item corresponds to one impression.

    Returns:
        exposure: (K,)
        click: (K,)
        semantic: (384,)
        history_embeddings: (L, 384)
        candidate_embeddings: (C, 384)
        label_index: scalar (index of clicked news in candidates)
        history_mask: 1 if user has history else 0
    """

    def __init__(self, split = "train", device = "cpu", max_history_len = 50):

        super().__init__()

        self.device = device
        self.max_history_len = max_history_len


        # Load behaviors
        if split == "train":
            behaviors_path = pv.TRAIN_BEHAVIORS_PATH
            news_path = pv.TRAIN_NEWS_PATH
        else:
            behaviors_path = pv.DEV_BEHAVIORS_PATH
            news_path = pv.DEV_NEWS_PATH

        self.behaviors_df = pd.read_csv(
            behaviors_path,
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


        # Load news dataframe (for categories)
        self.news_df = pd.read_csv(
            news_path,
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

        # Load category index
        with open(pv.CATEGORY_INDEX_PATH, "rb") as f:
            self.category_index = pickle.load(f)

        # Load news embeddings
        self.news_embeddings = np.load(pv.NEWS_EMBEDDINGS_PATH)
        with open(pv.NEWS_ID_TO_INDEX_PATH, "rb") as f:
            self.news_id_to_index = pickle.load(f)

        # Convert embeddings to tensor
        self.news_embeddings = torch.tensor(
            self.news_embeddings,
            dtype=torch.float32,
            device=device
        )

        # Attribute Builder
        self.attribute_builder = AttributeBuilder(
            news_df=self.news_df,
            category_index=self.category_index,
            news_embeddings={
                nid: self.news_embeddings[idx]
                for nid, idx in self.news_id_to_index.items()
            },
            device=device,
            verbose=False
        )

    def __len__(self):
        return len(self.behaviors_df)

    def __getitem__(self, idx):

        row = self.behaviors_df.iloc[idx]

        history_str = row["history"]
        impressions_str = row["impressions"]

        # Build Attributes
        attr_dict = self.attribute_builder.build_from_impression(impressions_str)

        exposure = attr_dict["exposure"]
        click = attr_dict["click"]
        semantic = attr_dict["semantic"]

        # History Embeddings
        history_embeddings = []

        if pd.isna(history_str):
            history_mask = torch.tensor(0.0, device=self.device)
        else:
            history_ids = history_str.split()

            # Truncate history to max length
            history_ids = history_ids[-self.max_history_len:]

            for nid in history_ids:
                if nid in self.news_id_to_index:
                    idx_emb = self.news_id_to_index[nid]
                    history_embeddings.append(
                        self.news_embeddings[idx_emb]
                    )

            if len(history_embeddings) > 0:
                history_embeddings = torch.stack(history_embeddings)
                history_mask = torch.tensor(1.0, device=self.device)
            else:
                history_embeddings = torch.zeros(
                    1, 384, device=self.device
                )
                history_mask = torch.tensor(0.0, device=self.device)

        # Candidate Embeddings + Label
        candidate_embeddings = []
        valid_labels = []
        items = impressions_str.split()

        for item in items:
            nid, label = item.split("-")

            if nid in self.news_id_to_index:
                idx_emb = self.news_id_to_index[nid]
                candidate_embeddings.append(self.news_embeddings[idx_emb])
                valid_labels.append(int(label))

   
        candidate_embeddings = torch.stack(candidate_embeddings)

        labels_tensor = torch.tensor(valid_labels, dtype=torch.long)

        positive_indices = torch.where(labels_tensor == 1)[0]

        if len(positive_indices) > 0:
            label_index = positive_indices[0]
        else:
            # If no click exists (rare case)
            label_index = torch.tensor(0)

        label_index = label_index.long()
        
        return (
            exposure,
            click,
            semantic,
            history_embeddings,
            candidate_embeddings,
            label_index,
            history_mask
        )