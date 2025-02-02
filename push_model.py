#!/usr/bin/env python3

#%%
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
from skops import hub_utils
from skops import card
from huggingface_hub import HfApi, create_repo
from yaml import load, Loader
import re

import importlib


print('got here')

#%%

def load_module(pyfile: str) -> int : 
    """
    Load a python module directly from a file.
    
    Args:
      pyfile (str): Path to the python module
    
    Returns:
      int: 0 if failed, 1 if successful
    """
    
    file_path = Path(pyfile)
    module_str = pyfile.split('/')[-1].replace(".py", "")
    
    parent = str(file_path.parent.resolve())
    
    if parent not in sys.path:
       sys.path.append(parent)
    
    try:
        globals()[module_str] = importlib.import_module(module_str) 
        return 1
    except Exception as e:
        print(e)
        return 0

#%%
def load_model(model_path):
    """
    Load a scikit-learn model and its metadata from a .skops file
    
    Args:
        model_path (str): Path to the .skops model file
    
    Returns:
        model: the loaded model
    """
    
    trusted_types = sio.get_untrusted_types(file = model_path)
    # print(trusted_types)
    model = sio.load(model_path, trusted = trusted_types)

    return model

#%%
def init_repo(model_path: str, local_repo: str,
              data: pd.DataFrame,
              requirements: list[str]=None):

    #parse and import dependencies    
    repo_reqs = []
    if requirements:
        for dep in requirements:
            module_name, namespace = dep.split(":")
            globals()[namespace] = importlib.import_module(namespace)
            req_str = "{0}={1}".format(module_name,
                                       getattr(globals()[namespace],'__version__'))
            repo_reqs.append(req_str)
    
    hub_utils.init(
        model = model_path,
        requirements = repo_reqs,
        dst = local_repo,
        task = config.get("model_card").get("task"),
        data = data
    )
    
#%%
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

def obtain_ytest_ypred():
    Xtest = pd.DataFrame(data.get('test'))   
    return data.get('test')[config['dataset']['target_col']], model.predict(Xtest)

def run_metrics(module: str, ytest, ypred):
    print(module)
    
    if module not in globals().keys():
        globals()[module] = importlib.import_module(f)
        
    if module == 'sklearn':
        metrics_module = getattr(globals()['sklearn'], 'metrics')
        
    else:
        metrics_module = globals()[module]
        
    print(metrics_module)
    
    metrics = config.get("model_card").get('metrics').get(module)    
    results = []
    
    for met in metrics:
        label, mfunc = met.split(":")
        
        kwargs = None
        
        if "(" in mfunc:
            kwargre = re.compile(r'((?:\w+=[\'\"]?\w+[\'\"]?)+)')
            kwargs = {k:eval(v) for (k,v) in map(lambda x: x.split("="), kwargre.findall(mfunc))}
            mfunc = mfunc.split("(")[0]
            
        res = {label:getattr(metrics_module, mfunc)(ytest, ypred, **kwargs if kwargs else {})}
        
        print(label, mfunc, kwargs)
        print(res)
        
        results.append(res)
        
    return results


def create_model_card(model: object, local_repo: str):
    model_card = card.Card(model,
                           metadata = card.metadata_from_config(Path(local_repo)))
    
    # model evaluation
    ytest, ypred = obtain_ytest_ypred()

    ## metrics
    for m in config.get("model_card").get("metrics").keys():
        results = run_metrics(m, ytest, ypred)
        for r in results:
            model_card.add_metrics(**r)
            

    return model_card

#%%
def main():
    parser = argparse.ArgumentParser(description="Push scikit-learn model to Hugging Face Hub")
    parser.add_argument('config_yaml', help='Path to yaml config file')
    # parser.add_argument("model_path", help="Path to .skops model file")
    # parser.add_argument("repo_name", help="Name of the Hugging Face repository")
    # parser.add_argument("--private", action="store_true", help="Create a private repository")
    # args = parser.parse_args()

    args = parser.parse_args()
    
    with open(args.config_yaml, 'r') as f:
        config = load(f, Loader)

    # import model dependencies before loading the model itself
    if config['model_deps']:    
        for pyfile in config['model_deps']:
            print(pyfile)
            load_module(pyfile)   

    # Load model and metadata
    model = load_model(config['model_path'])
    
    #load dataset
    if config['dataset'].get("source") == 'datasets':
        from datasets import load_dataset
        data = load_dataset(config['dataset'].get('name'))
    
    elseif config['dataset'].get('source') == 'csv':
        raise NotImplementedError("should be possible to point to csv file")
    
    #initialize local repo
    
    ## setup local repo directory
    ## temporary directory
    is_temp_repo = config.get('local_repo').get('name') == 'tmp'
    if is_temp_rep:
        import tempfile
        repo_dir = tempfile.TemporaryDirectory()
        local_repo = tmp_dir.name
    
    else:
        repo_dir = Path(config.get('local_repo').get('name'))
        local_repo = config.get('local_repo').get('name')
        
      
    
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
    print("got past mainguard")
    main()
