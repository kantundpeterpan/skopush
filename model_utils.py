import json
from pathlib import Path
import skops.io as sio

def validate_model_file(model_path):
    """
    Validate that the model file exists and is a valid .skops file
    
    Args:
        model_path (str): Path to the model file
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if path.suffix != ".skops":
        raise ValueError(f"Invalid file format. Expected .skops file, got: {path.suffix}")
    return True

def extract_model_metadata(model_path):
    """
    Extract metadata from a .skops model file
    
    Args:
        model_path (str): Path to the model file
    
    Returns:
        dict: Model metadata
    """
    metadata = sio.get_metadata(model_path)
    return {
        "name": metadata.get("name", ""),
        "description": metadata.get("description", ""),
        "sklearn_version": metadata.get("sklearn_version", ""),
        "python_version": metadata.get("python_version", ""),
        "metrics": metadata.get("metrics", {}),
        "class_labels": metadata.get("class_labels", []),
        "feature_names": metadata.get("feature_names", [])
    }

def save_metadata(metadata, output_path="metadata.json"):
    """
    Save metadata to a JSON file
    
    Args:
        metadata (dict): Model metadata
        output_path (str): Path to save the metadata
    """
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
