# Workflow scripts

This directory contains some scripts to execute the respective workflows. All scripts are meant to be executed from the root folder.

## Train models 

### Multi-Head Deep Label Distribution Learning for plant disease severity and age prediction

```bash
$ python scripts/train_multi_dldl.py -c config/models/<config>.yml
```
further information can be found in the help text by typing
```bash
$ python scripts/train_multi_dldl.py --help
```
