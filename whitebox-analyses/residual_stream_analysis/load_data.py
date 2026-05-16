from datasets import load_dataset
from dotenv import load_dotenv
import itertools

load_dotenv()

print("Loading dataset...", flush=True)
ds = load_dataset("uzaymacar/math-rollouts", streaming=True, split='default')

# print("Dataset loaded. Getting first few rows (no filter)...", flush=True)
# for i, row in enumerate(itertools.islice(ds, 3)):  # Just get first 3 items
#     print(f"\n=== Row {i} ===")
#     print(f"Keys: {list(row.keys())}")
#     print(f"Path value: {row.get('path', 'N/A')}")
#     if i >= 2:
#         break

print("\nNow trying with 'problem' filter...", flush=True)
filtered_ds = ds.filter(lambda example: "problem_1923" in example.get("path", ""))
for i, row in enumerate(itertools.islice(filtered_ds, 2)):
    print(f"Found: {row.get('path', 'N/A')}")