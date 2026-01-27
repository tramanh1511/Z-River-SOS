import pandas as pd
import re
import glob
import os
from collections import defaultdict

def parse_metrics_file(filepath, beta, data):
    data['alpha'].append(beta)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        # Use regex to split by multiple spaces
        parts = line.split(' ')
        metric, name, value = parts[0].strip(), " ".join(parts[1:-1]).strip(), float(parts[-1])
        data[name].append(value)
        # else skip

if __name__ == "__main__":
    input_folder= "data/vis_v2/2019_free_v2"
    output_path = "data/vis_v2/2019_free_v2/ablation_alpha.csv"
    # files = glob.glob(input_pattern)
    data = defaultdict(list)
    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)
        if not os.path.isfile(path):
            continue
        if 'macro' not in file:
            continue
        parts = file.split("_")
        print(parts)
        beta = parts[-1][:-4]
        print(beta)
        parse_metrics_file(path, float(beta), data)

    for key in data.keys():
        # Convert all values to float
        print(key, len(data[key]))
    df = pd.DataFrame(data)
    df = df.sort_values(by='alpha', ascending=True)
    print(df)
    df.iloc[:, :8].to_csv(output_path, index=False)  # Excel sheet names max 31 chars

    print(f"Exported all macro_*.txt files to {output_path}")