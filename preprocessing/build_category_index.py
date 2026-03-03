import pandas as pd
import pickle
import path_variables as pv


def load_news(path):
    return pd.read_csv(
        path,
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


def build_category_index():
    print("=" * 60)
    print("Building Category Index (TRAIN + DEV)")
    print("=" * 60)

    # Load train and dev news
    train_news = load_news(pv.TRAIN_NEWS_PATH)
    dev_news = load_news(pv.DEV_NEWS_PATH)

    print(f"Train categories: {train_news['category'].nunique()}")
    print(f"Dev categories: {dev_news['category'].nunique()}")

    # Combine and take UNION
    all_news = pd.concat([train_news, dev_news])
    categories = sorted(all_news["category"].unique())

    print("\nFinal Category List:")
    print(categories)
    print(f"\nTotal Categories (Union): {len(categories)}")

    # Build mapping
    category_index = {cat: idx for idx, cat in enumerate(categories)}

    # Save to processed folder
    with open(pv.CATEGORY_INDEX_PATH, "wb") as f:
        pickle.dump(category_index, f)

    print("\nSaved category_index.pkl at:")
    print(pv.CATEGORY_INDEX_PATH)


if __name__ == "__main__":
    build_category_index()