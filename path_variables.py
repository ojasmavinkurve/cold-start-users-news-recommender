import os

# ==========================================================
# SELECT DATASET HERE
# ==========================================================

DATASET_NAME = "MINDsmall"   # Change to "MINDlarge" later


# ==========================================================
# BASE DIRECTORIES
# ==========================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data", DATASET_NAME)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
DEV_DIR = os.path.join(DATA_DIR, "dev")
TEST_DIR = os.path.join(DATA_DIR, "test")  # may not exist for MINDsmall


# ==========================================================
# RAW FILE PATHS
# ==========================================================

TRAIN_NEWS_PATH = os.path.join(TRAIN_DIR, "news.tsv")
TRAIN_BEHAVIORS_PATH = os.path.join(TRAIN_DIR, "behaviors.tsv")

DEV_NEWS_PATH = os.path.join(DEV_DIR, "news.tsv")
DEV_BEHAVIORS_PATH = os.path.join(DEV_DIR, "behaviors.tsv")

TEST_NEWS_PATH = os.path.join(TEST_DIR, "news.tsv")
TEST_BEHAVIORS_PATH = os.path.join(TEST_DIR, "behaviors.tsv")


# ==========================================================
# PROCESSED OUTPUT PATHS
# ==========================================================

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

NEWS_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "news_embeddings.npy")
NEWS_ID_TO_INDEX_PATH = os.path.join(PROCESSED_DIR, "news_id_to_index.pkl")

TRAIN_ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, "train_attributes.pkl")
DEV_ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, "dev_attributes.pkl")

TRAIN_DATASET_CACHE = os.path.join(PROCESSED_DIR, "train_dataset.pkl")
DEV_DATASET_CACHE = os.path.join(PROCESSED_DIR, "dev_dataset.pkl")


# ==========================================================
# CREATE PROCESSED FOLDER IF NOT EXISTS
# ==========================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)