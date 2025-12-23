import pandas as pd
import json

# List of your JSONL files
jsonl_files = [
    "gnn_results/gat/gat_result_part_1.jsonl",
    "gnn_results/gat/gat_result_part_2.jsonl",
    "gnn_results/gat/gat_result_part_3.jsonl",
    "gnn_results/gat/gat_result_part_4.jsonl",
]

all_results = []
for file in jsonl_files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            all_results.append(json.loads(line))

df = pd.DataFrame(all_results)

df.to_csv("gnn_results/gat/gat_results_combined.csv", index=False)
print("All results saved to sage_results_combined.csv")
