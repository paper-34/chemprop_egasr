import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from chemprop.train.predict import predict
from chemprop.train.predict_and_emb import predict_and_emb
from chemprop.args import ExplainArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit

import torch

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import pickle 
import ot

import os

import matplotlib.pyplot as plt

@timeit()
def make_explanations(args: ExplainArgs, smiles: List[List[str]] = None):
    """
    Loads data and a trained model and uses the model to make explanations on the data.

    If SMILES are provided, then makes explanations on smiles.
    Otherwise makes explanations on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.ExplainArgs` object containing arguments for
                 loading data and a model and making explanations.
    :param smiles: List of list of SMILES to make explanations on.
    :return: A list of lists of target explanations.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # If features were used during training, they must be used when predicting
    if ((train_args.features_path is not None or train_args.features_generator is not None)
            and args.features_path is None
            and args.features_generator is None):
        raise ValueError('Features were used during training so they must be specified again during explanation '
                         'using the same type of features as before (with either --features_generator or '
                         '--features_path and using --no_features_scaling if applicable).')

    # Update explain args with training arguments to create a merged args object
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[ExplainArgs, TrainArgs]

    # Read train data
    full_data = get_data(path=args.test_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                         args=args, store_row=True)
    
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    exp_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    full_data = get_data(path=args.input_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                         args=args, store_row=True)
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    train_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])
    
    # Read test data
    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, store_row=True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])
    
    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    
    checkpoint_paths = args.checkpoint_paths
        
    def preds_and_embds(test_data):
        sum_preds = np.zeros((len(test_data), num_tasks))
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        sum_embds = {}
        
        for i, checkpoint_path in enumerate(tqdm(checkpoint_paths, total=len(checkpoint_paths))):
            # Load model and scalers
            model = load_checkpoint(checkpoint_path)
            scaler, features_scaler = load_scalers(checkpoint_path)

            # Normalize features
            if args.features_scaling:
                test_data.reset_features_and_targets()
                test_data.normalize_features(features_scaler)

            # Make predictions
            model_preds, model_embds = predict_and_emb(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler
            )
            sum_preds += np.array(model_preds)
            
            data_idx = 0
            if i==0:
                for batch_embds in model_embds:
                    graph_embds, a_scope = batch_embds[0]
                    for scope in a_scope:
                        sum_embds[data_idx] = graph_embds[scope[0]:(scope[0]+scope[1])]
                        data_idx+=1
            else:
                for batch_embds in model_embds:
                    graph_embds, a_scope = batch_embds[0]
                    for scope in a_scope:
                        sum_embds[data_idx] += graph_embds[scope[0]:(scope[0]+scope[1])]
                        data_idx+=1
        
        avg_preds = sum_preds / len(checkpoint_paths)
        
        avg_embds = {}
        for key, value in sum_embds.items():
            avg_embds[key] = value/len(checkpoint_paths)
        
        return avg_preds, avg_embds
    
    exp_preds, exp_embds = preds_and_embds(exp_data)
    train_preds, train_embds = preds_and_embds(train_data)
    
    # mu and sigma of the predicted properties of training data
    mu = np.mean(train_preds)
    sigma = np.std(train_preds)
    exp_sign = np.sign(exp_preds-mu) / sigma
    print(mu, sigma)
    
    train_preds_norm = (train_preds - mu) / sigma
    
    # Explanation by EGSAR
    def explain(graph_num, cont_embds, cont_preds, reg):
        cur_embds = exp_embds[graph_num].cpu().detach().numpy()
        cur_sign = 1
        
        cont_embds = [embds for embds in cont_embds.values()]

        w = []

        for i in range(len(cont_embds)):
            cont_embd = cont_embds[i].cpu().detach().numpy()
            x_weight = np.ones(cur_embds.shape[0])/cur_embds.shape[0]
            y_weight = np.ones(cont_embd.shape[0])/cont_embd.shape[0]
            M = ot.dist(cur_embds, cont_embd)
            w.append(-cur_sign * cont_preds[i] * np.sum(ot.sinkhorn(x_weight, y_weight, M/M.max(), reg) * M, axis=1))
            
        return np.sum(w, axis=0)
    
    w_all = []
    
    for i in range(len(exp_data)):
        w = explain(i, train_embds, train_preds_norm, args.exp_reg)
        w_all.append(w)

    # Save explanations
    print(f'Saving explanations to {args.exp_path}')
    makedirs(args.exp_path, isfile=True)

    # Error rasie for a classification mode
    if args.dataset_type == 'multiclass':
        raise ValueError('Now, EGSAR is not supported to multi-class data')

    # Save explanations
    with open(os.path.join(args.exp_path,'weights.pkl'), "wb") as fw:
        pickle.dump(w_all, fw)
        
    for i, ed in enumerate(exp_data):
        smiles = ed.smiles[0]

        image = SimilarityMaps.GetSimilarityMapFromWeights(ed.mol[0],w_all[i])
        image.savefig(os.path.join(args.exp_path, smiles + '_' + str(np.round(exp_preds[i],2)[0])  + '.png'), bbox_inches='tight')
        image.clf()

def egsar() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_explanations(args=ExplainArgs().parse_args())
