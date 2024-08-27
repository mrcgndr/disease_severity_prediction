# disease severity prediction

Supplementary code material for the research article

**SugarViT - Multi-objective regression of UAV images with Vision Transformers and Deep Label Distribution Learning demonstrated on disease severity prediction in sugar beet**

currently under review at PLOS One

Preprint available at [arXiv](https://arxiv.org/abs/2311.03076/)

## Installation

### 1. Clone and enter directory

```bash
$ git clone https://github.com/mrcgndr/disease_severity_prediction
$ cd phytopy
```

### 2. Create new Python environment and activate it

Tested with Python 3.11
```bash
$ python -m venv .venv
$ source .venv/bin/activate
```

### 3. Install requirements and package

Update pip
```bash
$ pip install -U pip
```

Install requirements
```bash
$ pip install -r requirements.txt
```

For installation as a regular package, run
```bash
$ pip install .
```

For installation in development mode, run
```bash
$ pip install -e .
```