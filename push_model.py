#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
from huggingface_hub import HfApi, create_repo

import sys
sys.path.append("/home/kantundpeterpan/projects")
from frugalai.tools import *

def load_model_and_metadata(model_path):
    """
    Load a scikit-learn model and its metadata from a .skops file
    
    Args:
        model_path (str): Path to the .skops model file
    
    Returns:
        tuple: (model, metadata) containing the loaded model and its metadata
    """
    
    trusted_types = sio.get_untrusted_types(file = model_path)
    print(trusted_types)
    model = sio.load(model_path, trusted = trusted_types)
    metadata = sio.get_metadata(model_path)
    return model, metadata

def create_confusion_matrix(y_true, y_pred, labels):
    """
    Create and save a confusion matrix plot
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        labels (list): List of class labels
    
    Returns:
        str: Path to saved confusion matrix image
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    
    output_path = "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_model_card(model_info, metrics, cm_path):
    """
    Generate a model card markdown file
    
    Args:
        model_info (dict): Model metadata
        metrics (dict): Model performance metrics
        cm_path (str): Path to confusion matrix image
    
    Returns:
        str: Path to generated model card
    """
    card_content = f"""---
tags:
- sklearn
- skops
- classification
library_name: sklearn
---

# Model Card for {model_info.get('name', 'Classification Model')}

## Model Description

{model_info.get('description', 'A scikit-learn classification model')}

## Intended Use & Limitations

This model is designed for classification tasks. Please refer to the metrics below for model performance.

## Training Procedure

The model was trained using scikit-learn version {model_info.get('sklearn_version', 'N/A')}.

## Metrics

```
{json.dumps(metrics, indent=2)}
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)
"""
    
    card_path = "README.md"
    with open(card_path, "w") as f:
        f.write(card_content)
    return card_path

def main():
    parser = argparse.ArgumentParser(description="Push scikit-learn model to Hugging Face Hub")
    parser.add_argument("model_path", help="Path to .skops model file")
    parser.add_argument("repo_name", help="Name of the Hugging Face repository")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    # Load model and metadata
    model, metadata = load_model_and_metadata(args.model_path)
    
    # Create repository
    api = HfApi()
    create_repo(args.repo_name, private=args.private, exist_ok=True)
    
    # Prepare files for upload
    files_to_upload = {
        "model.skops": args.model_path,
        "requirements.txt": "requirements.txt",
        "README.md": "README.md"
    }
    
    # Generate and save confusion matrix
    if hasattr(metadata, "test_data"):
        cm_path = create_confusion_matrix(
            metadata.test_data["y_true"],
            metadata.test_data["y_pred"],
            metadata.class_labels
        )
        files_to_upload["confusion_matrix.png"] = cm_path
    
    # Generate model card
    model_info = {
        "name": metadata.get("name", "Classification Model"),
        "description": metadata.get("description", ""),
        "sklearn_version": metadata.get("sklearn_version", "")
    }
    metrics = metadata.get("metrics", {})
    card_path = generate_model_card(model_info, metrics, "confusion_matrix.png")
    
    # Upload files to Hugging Face
    for dest, src in files_to_upload.items():
        api.upload_file(
            path_or_fileobj=src,
            path_in_repo=dest,
            repo_id=args.repo_name
        )

if __name__ == "__main__":
    main()
