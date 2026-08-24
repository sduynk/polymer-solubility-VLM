import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Paths
input_file = Path("dataset_resized_v2/_annotations.train.jsonl")
output_dir = Path("split_dataset")
output_dir.mkdir(exist_ok=True)

# Read the data
with input_file.open("r") as f:
    data = [json.loads(line) for line in f]

# Group by class using image name
class_groups = defaultdict(list)
for entry in data:
    image_name = entry["image"].lower()
    if "insoluble" in image_name:
        class_groups["insoluble"].append(entry)
    elif "partialsoluble" in image_name or "partialsobule" in image_name:
        class_groups["partialsoluble"].append(entry)
    elif "colloidal" in image_name:
        class_groups["colloidal"].append(entry)
    elif "soluble" in image_name:
        class_groups["soluble"].append(entry)


# Shuffle and split each class
splits = {"train": [], "valid": [], "test": []}
for cls, items in class_groups.items():
    random.shuffle(items)
    n = len(items)
    n_train = int(0.7 * n)
    n_valid = int(0.15 * n)
    n_test = n - n_train - n_valid
    splits["train"].extend(items[:n_train])
    splits["valid"].extend(items[n_train:n_train + n_valid])
    splits["test"].extend(items[n_train + n_valid:])

# Write to output files
for split_name, split_data in splits.items():
    with (output_dir / f"_annotations.{split_name}.jsonl").open("w") as f:
        for item in split_data:
            f.write(json.dumps(item) + "\n")

split_summary = {
    split: {
        "total": len(items),
        "insoluble": sum("insoluble" in e["image"] for e in items),
        "soluble": sum("soluble" in e["image"] for e in items),
        "partialsoluble": sum("partialsoluble" in e["image"] or "partialsobule" in e["image"] for e in items),
        "colloidal": sum("colloidal" in e["image"] for e in items),
    }
    for split, items in splits.items()
}

df_summary = pd.DataFrame(split_summary).T
print(df_summary)
df_summary.to_csv("dataset_split_summary.csv", index=False)
