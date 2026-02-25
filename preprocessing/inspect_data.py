import os
import pandas as pd
from collections import Counter
import path_variables as pv

# ==============================
# CONFIG
# ==============================
TRAIN_PATH = pv.TRAIN_DIR
DEV_PATH = pv.DEV_DIR


# ==============================
# UTILITY FUNCTIONS
# ==============================

def inspect_news(news_file):

    print("\n" + "="*50)
    print("Inspecting news.tsv")
    print("="*50)

    news_df = pd.read_csv(
        news_file,
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
        ]
    )

    print(f"Total News Articles: {len(news_df)}")

    unique_categories = news_df["category"].nunique()
    print(f"Unique Categories: {unique_categories}")

    category_list = sorted(news_df["category"].unique())
    print("\nAll Categories:")
    print(category_list)

    category_counts = news_df["category"].value_counts()
    print("\nTop 10 Categories:")
    print(category_counts.head(10))

    missing_titles = news_df["title"].isnull().sum()
    print(f"\nMissing Titles: {missing_titles}")

    return news_df


def inspect_behaviors(behaviors_file):
    print("\n" + "="*50)
    print("Inspecting behaviors.tsv")
    print("="*50)

    behaviors_df = pd.read_csv(
        behaviors_file,
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

    print(f"Total Impressions: {len(behaviors_df)}")

    unique_users = behaviors_df["user_id"].nunique()
    print(f"Unique Users: {unique_users}")

    # --------------------------
    # History statistics
    # --------------------------

    history_lengths = []
    zero_history_users = 0

    for hist in behaviors_df["history"]:
        if pd.isna(hist):
            history_lengths.append(0)
            zero_history_users += 1
        else:
            length = len(hist.split())
            history_lengths.append(length)
            if length == 0:
                zero_history_users += 1

    avg_history = sum(history_lengths) / len(history_lengths)

    print(f"Average History Length: {avg_history:.2f}")
    print(f"Users with Zero History: {zero_history_users}")

    max_history = max(history_lengths)
    print(f"Max History Length: {max_history}")

    # --------------------------
    # Candidate statistics
    # --------------------------

    candidate_lengths = []
    click_counts = []

    for imp in behaviors_df["impressions"]:
        items = imp.split()
        candidate_lengths.append(len(items))

        # count clicks
        clicks = sum([1 for item in items if item.endswith("-1")])
        click_counts.append(clicks)

    avg_candidates = sum(candidate_lengths) / len(candidate_lengths)
    avg_clicks = sum(click_counts) / len(click_counts)

    print(f"\nAverage Candidate News per Impression: {avg_candidates:.2f}")
    print(f"Average Clicks per Impression: {avg_clicks:.2f}")

    click_distribution = Counter(click_counts)
    print("\nClick Count Distribution (clicks per impression):")
    for k in sorted(click_distribution.keys()):
        print(f"{k} clicks: {click_distribution[k]} impressions")

    return behaviors_df


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("\nInspecting TRAIN split")
    train_news = inspect_news(pv.TRAIN_NEWS_PATH)
    train_behaviors = inspect_behaviors(pv.TRAIN_BEHAVIORS_PATH)

    print("\nInspecting DEV split")
    dev_news = inspect_news(pv.DEV_NEWS_PATH)
    dev_behaviors = inspect_behaviors(pv.DEV_BEHAVIORS_PATH)

    print("\nInspection Complete.")