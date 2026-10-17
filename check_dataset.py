
from pathlib import Path

dataset_path = Path("data/frankenstein.txt")

if not dataset_path.exists():
    print("ERROR: Dataset not found!")
    print("Expected location:", dataset_path.resolve())
else:
    with open(dataset_path, "r", encoding="utf-8") as file:
        text = file.read()

    words = text.split()

    print("Dataset loaded successfully!")
    print("File size:", round(dataset_path.stat().st_size / 1024, 2), "KB")
    print("Total characters:", len(text))
    print("Total whitespace-separated tokens:", len(words))
    print("\nFirst 500 characters:\n")
    print(text[:500])
