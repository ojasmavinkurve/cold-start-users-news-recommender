from cProfile import label

import torch
import pandas as pd
import os
from collections import Counter
from typing import Dict, List


class AttributeBuilder:
    """
    Builds raw attribute feature vectors from a single impression.

    Prints:
        - Parsed news IDs
        - Exposure vector
        - Click vector
        - Semantic prior
        - Final concatenated raw attribute vector
    """

    def __init__(
        self,
        news_df: pd.DataFrame,
        category_index: Dict[str, int],
        news_embeddings: Dict[str, torch.Tensor],
        device: str = "cpu",
        verbose: bool = True
    ):
        self.news_df = news_df.set_index("news_id")
        self.category_index = category_index
        self.news_embeddings = news_embeddings
        self.num_categories = len(category_index)
        self.device = device
        self.verbose = verbose
        self.news_to_category = dict(zip(news_df["news_id"], news_df["category"]))

    def _parse_impression(self, impressions: str):
        news_ids = []
        clicked_ids = []

        for item in impressions.split():
            nid, label = item.split("-")
            news_ids.append(nid)
            #removed clicked impressions to avoid overfitting
            if label == "1":
                clicked_ids.append(nid)

        return news_ids, clicked_ids


    def compute_exposure_vector(self, news_ids: List[str]) -> torch.Tensor:

        exposure_vec = torch.zeros(self.num_categories, device=self.device)

        if len(news_ids) == 0:
            return exposure_vec

        categories = [
            self.news_to_category[nid]
            for nid in news_ids
            if nid in self.news_to_category   
        ]

        counter = Counter(categories)
        total = len(categories)

        for cat, count in counter.items():
            if cat in self.category_index:
                exposure_vec[self.category_index[cat]] = count / total

        if self.verbose:
            print("Exposure Vector:", exposure_vec)

        return exposure_vec


    def compute_click_vector(self, clicked_ids: List[str]) -> torch.Tensor:

        click_vec = torch.zeros(self.num_categories, device=self.device)

        if len(clicked_ids) == 0:
            if self.verbose:
                print("Click Vector: ZERO (No clicks)")
            return click_vec

        categories = [
            self.news_to_category[nid]
            for nid in clicked_ids
            if nid in self.news_to_category
        ]
        counter = Counter(categories)
        total = len(clicked_ids)

        for cat, count in counter.items():
            if cat in self.category_index:
                click_vec[self.category_index[cat]] = count / total

        if self.verbose:
            print("Click Vector:", click_vec)

        return click_vec


    def compute_semantic_prior(self, clicked_ids: List[str]) -> torch.Tensor:

        if len(clicked_ids) == 0:
            semantic = torch.zeros(384, device=self.device)
            if self.verbose:
                print("Semantic Prior: ZERO (No clicks)")
            return semantic

        vectors = [
            self.news_embeddings[nid]
            for nid in clicked_ids
            if nid in self.news_embeddings
        ]

        if len(vectors) == 0:
            semantic = torch.zeros(384, device=self.device)
            if self.verbose:
                print("Semantic Prior: ZERO (No embeddings found)")
            return semantic

        stacked = torch.stack(vectors)
        semantic = stacked.mean(dim=0)

        if self.verbose:
            print("Semantic Prior Vector (first 10 dims):", semantic[:10])

        return semantic

    def build_from_impression(self, impressions: str):

        news_ids, clicked_ids = self._parse_impression(impressions)

        #remove clicked items from exposure
        non_clicked_ids = [
            nid for nid in news_ids
            if nid not in clicked_ids
        ]

        exposure_vec = self.compute_exposure_vector(non_clicked_ids)

        click_vec = self.compute_click_vector(clicked_ids)
        semantic_vec = self.compute_semantic_prior(clicked_ids)

        return {
            "exposure": exposure_vec,
            "click": click_vec,
            "semantic": semantic_vec
        }
    