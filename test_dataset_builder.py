import torch
import random
from datasets.dataset_builder import MindDataset


def test_dataset():

    print("=" * 60)
    print("Testing MindDataset")
    print("=" * 60)

    # Create dataset
    dataset = MindDataset(
        split="train",
        device="cpu",
        max_history_len=50
    )

    print(f"Total impressions: {len(dataset)}")

    # Get one sample
    idx = random.randint(0, len(dataset)-1)
    sample = dataset[idx]

    (
        exposure,
        click,
        semantic,
        history_embeddings,
        candidate_embeddings,
        label_index,
        history_mask
    ) = sample

    print("\n--- Sample Shapes ---")
    print("Exposure shape:", exposure.shape)
    print("Click shape:", click.shape)
    print("Semantic shape:", semantic.shape)
    print("History embeddings shape:", history_embeddings.shape)
    print("Candidate embeddings shape:", candidate_embeddings.shape)
    print("Label index:", label_index)
    print("History mask:", history_mask)

    print("\nTest complete.")


if __name__ == "__main__":
    test_dataset()