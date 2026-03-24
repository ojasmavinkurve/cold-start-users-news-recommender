# TO do: pip install sentence-transformers

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
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


def build_news_embeddings(batch_size=64):
    print("=" * 60)
    print("Building News Embeddings using MiniLM")
    print("=" * 60)

    # Load train + dev news
    train_news = load_news(pv.TRAIN_NEWS_PATH)
    dev_news = load_news(pv.DEV_NEWS_PATH)
    test_news = load_news(pv.TEST_NEWS_PATH)

    print(f"Train news: {len(train_news)}")
    print(f"Dev news: {len(dev_news)}")
    print(f"Test news: {len(test_news)}")

    # Combine & remove duplicates
    all_news = pd.concat([train_news, dev_news, test_news])
    all_news = all_news.drop_duplicates(subset=["news_id"])

    print(f"Total unique news articles: {len(all_news)}")

    # Load MiniLM model
    print("Loading MiniLM model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode titles
    titles = all_news["title"].fillna("").tolist()

    print("Encoding titles...")
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    # Create news_id → index mapping
    news_id_list = all_news["news_id"].tolist()
    news_id_to_index = {news_id: idx for idx, news_id in enumerate(news_id_list)}

    # Save embeddings
    np.save(pv.NEWS_EMBEDDINGS_PATH, embeddings)

    with open(pv.NEWS_ID_TO_INDEX_PATH, "wb") as f:
        pickle.dump(news_id_to_index, f)

    print("\nSaved files:")
    print(pv.NEWS_EMBEDDINGS_PATH)
    print(pv.NEWS_ID_TO_INDEX_PATH)
    print("\nDone.")


if __name__ == "__main__":
    build_news_embeddings()