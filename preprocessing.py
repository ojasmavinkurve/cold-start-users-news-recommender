import os

print("Starting preprocessing pipeline...\n")

# Step 1: Build category index
print("Building category index...")
os.system("python -m preprocessing.build_category_index")

# Step 2: Build news embeddings
print("\nBuilding MiniLM news embeddings...")
os.system("python -m preprocessing.build_news_embeddings")

print("\nPreprocessing completed successfully.")