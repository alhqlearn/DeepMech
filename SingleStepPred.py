

import os, sys
import numpy as np
import pandas as pd
sys.path.append('scripts')
from rdkit.Chem import PandasTools
from Synthesis import init_DeepMech, predict_product
import heapq
from tqdm import tqdm
import torch
import copy
from rdkit import Chem
import ast
import math
import csv
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
from tqdm import tqdm
from my_inference_utils import  process_results



def model_predict(reactants):
    sep = False
    verbose = False
    results_df, results_dict = predict_product(args, reactants, modelx, graph_functions,
                                           template_dicts, template_infos,
                                           verbose = verbose, sep = sep, collect_n = 100)

    #adding the heuristic here:
    results_dict = process_results(results_dict)
    products = []
    scores = []
    for key, value in results_dict.items():
        if key[:3] == 'Top':
            only_product = results_dict[key]['product']
            products.append(only_product)
            scores.append(results_dict[key]['score'])
    output = [(i, j) for i, j in zip(products[:5], scores[:5])]
    output = sorted(output, key=lambda x:x[1], reverse=True)
    return output

'''
reactants = 'CC(N)C([O-])=O.C/C=[NH+]/C'

output = model_predict(reactants)
print('output:', output)
'''

def canonicalize(smiles):
    """Canonicalize a SMILES string, removing stereochemistry."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove stereochemistry
        #mol = rdMolStandardize.Cleanup(mol)
        #Chem.RemoveStereochemistry(mol)
        return MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def demap_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def calculate_top_k_accuracies(ground_truth, predictions, max_k=5):
    """
    Calculate top-1 to top-k accuracy.
    :param ground_truth: Ground truth SMILES string.
    :param predictions: List of predicted tuples (SMILES, score).
    :param max_k: Maximum k-value to compute accuracy for.
    :return: List of top-k accuracy values.
    """
    ground_truth_canonical = canonicalize(demap_smiles(ground_truth))
    accuracies = [0] * max_k  # Initialize accuracy list with zeros

    if ground_truth_canonical is None:
        return accuracies  # If ground truth is invalid, return 0s

    # Canonicalize top predictions
    top_predictions = [canonicalize(prod[0]) for prod in predictions[:max_k] if prod[0] is not None]

    # Compute accuracy for Top-1, Top-2, ..., Top-k
    for i in range(max_k):
        if ground_truth_canonical in top_predictions[:i+1]:  # Check if it's in the first (i+1) predictions
            accuracies[i] = 1  # Mark as hit
        
    return accuracies


def process_file(file_path, output_csv, max_k=5):
    """
    Process the input file to predict and calculate top-k accuracies.

    :param file_path: Path to the input .txt file with reaction pairs.
    :param output_csv: Path to the output CSV file for saving predictions.
    :param max_k: Maximum k-value for computing accuracy.
    """
    all_predictions = []
    accuracy_counters = [0] * max_k  # Track how many times each Top-k is hit
    total_samples = 0
    results_to_save = []

    # Read the input file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    from tqdm import tqdm
    for line in tqdm(lines):
        reactants, ground_truth = line.strip().split('>>')
        reactants = demap_smiles(reactants)

        # Predict products
        predictions = model_predict(reactants)

        # Save the predictions
        for product, score in predictions:
            all_predictions.append([reactants, product, score])

        # Calculate Top-1 to Top-k accuracy
        top_k_accuracies = calculate_top_k_accuracies(ground_truth, predictions, max_k)

        # Update counters for overall accuracy
        for i in range(max_k):
            accuracy_counters[i] += top_k_accuracies[i]

        total_samples += 1

        # Prepare results to save for this line
        results_to_save.append({
            'Reactants': reactants,
            'Ground_Truth': ground_truth,
            'Top-1 Accuracy': top_k_accuracies[0],
            'Top-2 Accuracy': top_k_accuracies[1],
            'Top-3 Accuracy': top_k_accuracies[2],
            'Top-4 Accuracy': top_k_accuracies[3],
            'Top-5 Accuracy': top_k_accuracies[4],
            'Predictions': str(predictions)
        })

    # Save all predictions to a CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Reactants', 'Ground_Truth', 'Top-1 Accuracy', 'Top-2 Accuracy', 'Top-3 Accuracy',
                      'Top-4 Accuracy', 'Top-5 Accuracy', 'Predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_to_save)

    # Calculate and print overall accuracy for each k
    overall_accuracies = [accuracy_counters[i] / total_samples for i in range(max_k)]
    for i in range(max_k):
        print(f"Overall Top-{i+1} Accuracy: {overall_accuracies[i]:.6f}")

    return all_predictions, overall_accuracies


# Example usage
import time
st_time = time.time()

#------------CHANGE HERE-----------------------------------------
dataset = 'ID_Split42' # get the info of derived templates
scenario = 'Split42'
#--------------------------------------------------------------
device = 'cuda' # cpu or cuda
model_name = 'DeepMech_%s' % scenario
model_path = 'models/%s.pth' % model_name
config_path = 'data/configs/default_config'
data_dir = 'data/%s' % dataset
args = {'data_dir': data_dir, 'model_path': model_path, 'config_path': config_path, 'device': device, 'mode': 'test'}
modelx, graph_functions, template_dicts, template_infos = init_DeepMech(args)
#-----------------------CHANGE HERE--------------------------
input_file = "./data/ID_Split/test.txt"  # Your .txt file containing reaction pairs
output_file = './data/ID_Split/SingleStepPredOut.csv' # Output CSV file to save predictions
#----------------------------------------------------------------
predictions, accuracy = process_file(input_file, output_file, max_k=5)
en_time = time.time()
print('Time required:', (en_time-st_time)/60)






