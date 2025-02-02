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
    
    cm_config = config['model_card']['confusion_matrix']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    
    plt.title("Confusion Matrix")
    
    #run any addition plt functions
    if 'plt' in cm_config.keys():
        for f in cm_config['plt']:
            getattr(plt, f)(**{k:eval(v) for (k,v) in map(lambda x:x.split("="), cm_config['plt'][f])})
    
    # save the plot
    output_path = Path(local_repo) / cm_config.get('filename', 'confusion_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')
    
    return output_path

def obtain_ytest_ypred():
    """
    Obtain the target values (y_test) and predicted values (y_pred) for the test dataset.

    This function retrieves the test dataset from a global `data` dictionary and extracts
    the target column specified in the global `config`. It then uses a global `model`
    to make predictions on the test dataset.

    Returns:
        tuple: A tuple containing:
            - y_test (pd.Series): The actual target values from the test dataset.
            - y_pred (np.ndarray): The predicted values from the model.
    """
    if config['dataset']['source'] == 'datasets':
        
        # Get evaluation set from config
        eval_set = config['dataset']['evaluate_on']
        
        # Get target column from config
        target_col = config['dataset']['target_col']
        
        # Retrieve the test dataset as a DataFrame
        Xtest = pd.DataFrame(data.get(eval_set))
        
        # Extract the target column and make predictions using the model
        return data.get(eval_set)[target_col], model.predict(Xtest)

    else:
        raise NotImplementedError('Need better logic for other dataset sources')



def run_metrics(module: str, ytest, ypred):
    """
    Dynamically run evaluation metrics on test data predictions using a specified module.

    This function dynamically imports a metrics module (e.g., sklearn), retrieves metrics
    defined in a global configuration, and computes them for given test labels (`ytest`)
    and predictions (`ypred`).

    Args:
        module (str): The name of the module containing metric functions (e.g., 'sklearn').
        ytest (array-like): The ground truth target values.
        ypred (array-like): The predicted values from a model.

    Returns:
        list: A list of dictionaries where each dictionary contains:
            - label (str): The name of the metric.
            - result: The computed value of the metric.
    """
    print(module)  # Debugging: Print the module name

    # Dynamically import the specified module if not already loaded
    if module not in globals().keys():
        globals()[module] = importlib.import_module(module)

    # Retrieve the metrics module based on whether it's 'sklearn' or another module
    if module == 'sklearn':
        metrics_module = getattr(globals()['sklearn'], 'metrics')
    else:
        metrics_module = globals()[module]

    print(metrics_module)  # Debugging: Print the loaded metrics module

    # Retrieve metric configurations from the global `config`
    metrics = config.get("model_card").get("metrics").get(module)
    
    # Initialize an empty list to store results
    results = []

    # Iterate through each metric defined in the configuration
    for met in metrics:
        # Split metric definition into label and function name
        label, mfunc = met.split(":")
        
        kwargs = None  # Initialize keyword arguments as None

        # Check if additional arguments are provided in parentheses within `mfunc`
        if "(" in mfunc:
            # Use regex to extract key-value pairs from parentheses
            kwargre = re.compile(r'(?:\w+=["\']?\w+["\']?)')
            kwargs = {k: eval(v) for k, v in map(lambda x: x.split("="), kwargre.findall(mfunc))}
            
            # Extract only the function name (before parentheses)
            mfunc = mfunc.split("(")[0]

        # Dynamically call the metric function with or without additional arguments
        res = {label: getattr(metrics_module, mfunc)(ytest, ypred, **kwargs if kwargs else {})}

        print(label, mfunc, kwargs)  # Debugging: Print details of each metric computation
        print(res)  # Debugging: Print computed result

        # Append result to results list
        results.append(res)

    return results  # Return all computed metric results



def create_model_card(model: object, local_repo: str):
    
    card_config = config['model_card']
    
    model_card = card.Card(model,
                           metadata = card.metadata_from_config(Path(local_repo)))
    
    # model description
    model_card.add(
        **{"Model description":card_config.get("description").get("main")}
    )
    
    ## check for subheadings of model description
    desc_config = config['model_card']['description']
    if len(desc_config.keys()) > 1: # 'main' must be included
        for k in [k for k in desc_config.keys() if k != 'main']:
            model_card.add(
                **{'/'.join(["Model description", k]):desc_config.get(k)}
            )
    
    
    # any other sections
    if 'sections' in card_config.keys():
        for s in card_config['sections']:
            model_card.add(
                **{s:card_config['sections'][s]}
            )
    
    # model evaluation
    ytest, ypred = obtain_ytest_ypred()

    ## metrics
    for m in config.get("model_card").get("metrics").keys():
        results = run_metrics(m, ytest, ypred)
        for r in results:
            model_card.add_metrics(**r)
            
    # confusion matrix
    cm_config = config['model_card']['confusion_matrix']
    ## create the confusion matrix, save to local repo folder
    cm_path = create_confusion_matrix(ytest, ypred, model.classes_)
    ## add plot to card
    model_card.add_plot(
        **{cm_config.get("title", "Confusion Matrix"):cm_config.get("filename", "confusion_matrix.png")}
    )

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
    
    global config, model, data, local_repo
    
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
    
    elif config['dataset'].get('source') == 'csv':
        raise NotImplementedError("should be possible to point to csv file")
    
    #initialize local repo
    
    ## setup local repo directory
    ## temporary directory
    is_temp_repo = config.get('local_repo').get('name') == 'tmp'
    
    if is_temp_repo:
        import tempfile
        repo_dir = tempfile.TemporaryDirectory()
        local_repo = repo_dir.name
    
    else:
        repo_dir = Path(config.get('local_repo').get('name'))
        local_repo = config.get('local_repo').get('name')
        
    # init local repo
    init_repo(config['model_path'], local_repo,
              data.get(config['dataset']['evaluate_on']),
              config['deps'])
    
    # create the model card
    mcard = create_model_card(model, local_repo)
    
    # save mdoel card to repo
    mcard.save(Path(local_repo) / "README.md")
    
    # add additional files
    for f in config['model_deps']:
        hub_utils.add_files(
            Path(f),
            dst = Path(local_repo)
        )
        
    # push skopush.yaml
    hub_utils.add_files(
        Path(args.config_yaml),
        dst = Path(local_repo)
    )
    
    # set create_remote to True if the repository doesn't exist remotely on the Hugging Face Hub
    hub_utils.push(
        repo_id=config['hf_repo'],
        source=local_repo,
        commit_message=config['push']['commit_message'],
        create_remote=config['push']['create_remote'],
        token = os.environ['HF_TOKEN']
    )
    
    if is_temp_repo:
        repo_dir.cleanup()
    
if __name__ == "__main__":
    print("got past mainguard")
    main()
