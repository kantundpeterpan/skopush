

# <img src="skopush_logo.png" style="border-radius:50%; max-width:10%" alt="skopush"> skopush

A tool for pushing scikit-learn models to the Hugging Face Hub with
automated model card generation.

## Features

- Automated model card generation
- Dynamic metric computation
- Confusion matrix visualization
- Flexible dependency management
- Support for datasets from Hugging Face hub
- Configurable through YAML files

## Installation

``` bash
pip install skops huggingface-hub numpy pandas matplotlib scikit-learn pyyaml
```

## Usage

The tool requires two main components: - A trained scikit-learn model
saved in .skops format - A configuration YAML file

To push a model to Hugging Face Hub:

``` bash
HF_TOKEN=... python skopush.py config.yaml
```

## Configuration

Create a YAML configuration file with the following structure:

``` yaml
hf_repo: "username/repository-name"
local_repo:
  name: "repository-path"
  init: true
model_path: "model-file.skops"
dataset:
  name: "dataset-name"
  source: "datasets"
  target_col: "target-column"
  evaluate_on: "test"
```

**Required Environment Variables** - `HF_TOKEN`: Your Hugging Face
authentication token

## Features in Detail

**Model Card Generation** - Automatically generates comprehensive model
cards - Includes model description, metrics, and visualizations -
Supports custom sections and metadata

**Metric Computation** - Dynamic computation of evaluation metrics -
Supports sklearn metrics out of the box - Extensible to custom metric
modules

**Visualization** - Automated confusion matrix generation - Configurable
plot styling - Support for custom matplotlib configurations

## Example Configuration

``` yaml
model_card:
  task: "text-classification"
  description:
    main: "Model description"
  metrics:
    sklearn:
      - "accuracy:accuracy_score"
      - "f1_score:f1_score(average='macro')"
  confusion_matrix:
    title: "Confusion Matrix"
    filename: "confusion_matrix.png"
```

## Dependencies

The tool manages both model dependencies and runtime dependencies: -
Model dependencies: Python files needed for model loading - Runtime
dependencies: Python packages required for execution

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open source and available under the MIT License.
