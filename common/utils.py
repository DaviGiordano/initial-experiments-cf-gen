import json
import matplotlib.pyplot as plt
import os

def generate_metadata(df, output_path):
    metadata = {}

    for col in df.columns:
        metadata[col] = {
            'dtype': str(df[col].dtype)
        }
        if df[col].dtype == 'object':
            metadata[col]['distinct_values'] = df[col].unique().tolist()

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
