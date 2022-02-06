# EGSAR
This repository contains the implementation of egsar to chemprop.

## Installing from source

1. `Download this source and change directory to the folder containing this source`
2. `conda env create -f environment.yml -n chemprop_egsar`
3. `conda activate chemprop_egsar`
4. `conda install -c conda-forge rdkit`
5. `pip install git+https://github.com/bp-kelley/descriptastorus`
6. `pip install -e .`
 
## Training (using chemprop)

For example, to train a model, run:
```
chemprop_train --data_path data/EGFR_train.csv --dataset_type regression --save_dir model_egfr
```

## Explanation (by EGSAR)

To load a trained model and make explanations, run `egsar.py` and specify:
* `--test_path <path>` Path to the data to explain.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--exp_path` Path where the pickle files for explanation and explained molecule images will be saved.
* `--exp_reg` Lambda for Sinkhorn algorithm.

For example:
```
egsar --test_path data/EGFR_test.csv --checkpoint_dir model_egfr --input_path data/EGFR_train.csv --exp_path exp_egfr/ --reg 0.01
```

## Key previosu works: Molecular Property Prediction (For D-MPNN implementation) & CoGE (Using contrastive learning)

**chemprop (D-MPNN):** https://github.com/chemprop/chemprop

**CoGE:** https://github.com/lukasjf/contrastive-gnn-explanation/tree/master/gnnexplainer
