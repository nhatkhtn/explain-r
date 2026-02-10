# EXPLAIN-R

This is the official implementation of the paper "[Advancing Interpretability of CLIP Representations with Concept Surrogate Model](https://openreview.net/pdf?id=KxoPiQ03BT)", published at NeurIPS'25.

## Install

The code was tested on Ubuntu with an NVIDIA A100. It requires at least Python 3.12 and ```transformers>=4.50.0```.

First, install the requirements (to your virutal environment):
```
pip install -r requirements.txt
```

Then, install this repo:
```
poetry install
```

## Configure

Before running the experiments, you need to create an ```.env``` file with the following keys:

```
WANDB_PROJECT=explain-representation
REPO_ROOT=<path to repo root>
COCO_ROOT=<path to downloaded COCO dataset>
IMAGENET_ROOT=<path to downloaded ImageNet dataset>
FLICKR30K_ROOT=<path to downloaded Flickr30k dataset>
DOWNLOAD_ROOT=<path to automatically store remaining downloaded datasets>
```

## Experiments

In the following instruction, we use the ImageNet dataset as the example. To use another dataset, simply replace ImageNet with (COCO, Flickr30k, SUN397, Flowers102, Food101).

To run this code on other datasets, define the dataset in ```exrep/dataset``` following existing examples.

### Step 0: obtain captions (for datasets without captions).

By default, we use the ```blip2-opt-2.7b-coco``` model. For example:

```python
python scripts/captioning.py ImageNet --split test --temperature 1.3
```
The above command would generate a file at (by default): ```outputs/imagenet/captions.json```.

### Step 1: Concept identification. Example:
```python
python scripts/concept_encode.py ImageNet -c outputs/imagenet/captions.json --max_threshold 0.1
```

The higher ```max_threshold``` is, the lower the sparsity. The above command generates a pickle file containing the concept vectors c_i at (by default): ```outputs/imagenet/average_encoding.pkl```.

### Step 2: Training the surrogate

See our [example](configs/train_imagenet.yaml) config file for the parameters.
```python
python scripts/train_surrogate.py --config config.yml
```

The checkpoints are saved by default at ```outputs/imagenet/ckpts/<run_name>_<>.pt```.

### Step 3: Obtain explanations

See `notebooks/attribution_single.ipynb`
