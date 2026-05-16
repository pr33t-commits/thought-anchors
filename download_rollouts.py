from datasets import load_dataset
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm

load_dotenv()

print("Loading dataset...", flush=True)
ds = load_dataset("uzaymacar/math-rollouts", streaming=True, split='default')

print("Filtering and creating files...", flush=True)
filtered_ds = ds.filter(lambda example: "problem_1591" in example.get("path", ""))

for i, row in tqdm(enumerate(filtered_ds)):
    
    file_path = os.path.join("math_rollouts",row.get("path"))
    content = row.get("content")
    
    if file_path and content:
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        if (i + 1) % 100 == 0:
            print(file_path, flush=True)
            print(f"Created {i + 1} files", flush=True)

print("Done!")