import pandas as pd
from sklearn.model_selection import train_test_split
import path_variables as pv

def split_train_dev(dev_ratio=0.1, seed=42):

    df = pd.read_csv(
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

    # -----------------------------------------
    # ✅ USER-BASED SPLIT
    # -----------------------------------------
    users = df["user_id"].unique()

    train_users, dev_users = train_test_split(
        users,
        test_size=dev_ratio,
        random_state=seed
    )

    train_df = df[df["user_id"].isin(train_users)]
    dev_df = df[df["user_id"].isin(dev_users)]

    # Save
    train_df.to_csv(pv.TRAIN_BEHAVIORS_PATH, sep="\t", header=False, index=False)
    dev_df.to_csv(pv.DEV_BEHAVIORS_PATH, sep="\t", header=False, index=False)

    print("Split complete!")
    print(f"Train users: {len(train_users)}")
    print(f"Dev users: {len(dev_users)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Dev rows: {len(dev_df)}")


if __name__ == "__main__":
    split_train_dev()