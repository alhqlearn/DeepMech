

#common functions used for inferencing

import os, sys
import numpy as np
import pandas as pd
sys.path.append('scripts')
from rdkit.Chem import PandasTools
import heapq
from tqdm import tqdm
import torch
import copy
from rdkit import Chem
import ast
import math
import ast
import pandas as pd

def compute_top_n_accuracy(df, top_n=5, true_col='true_mech', pred_prefix='top', verbose=False):
    """
    Compute Top-n accuracy for list-valued predictions in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing ground truth and top-k predictions
    - top_n: maximum k for top-k accuracy (default: 5)
    - true_col: column name for ground truth (default: 'true_mech')
    - pred_prefix: prefix for prediction columns (e.g., 'top' ? 'top1', 'top2', ...)
    - verbose: if True, prints errors and progress info
    
    Returns:
    - A dictionary { 'top1': acc1, 'top2': acc2, ..., 'topN': accN }
    """
    
    top_n_hits = {f'{pred_prefix}{k}': 0 for k in range(1, top_n + 1)}
    total = 0

    for idx, row in df.iterrows():
        try:
            if pd.isna(row[true_col]):
                continue

            true_list = ast.literal_eval(row[true_col])
            true_list = canonicalize_smiles(true_list)
            total += 1

            for k in range(1, top_n + 1):
                pred_col = f'{pred_prefix}{k}'

                if pred_col not in df.columns or pd.isna(row[pred_col]):
                    continue

                pred_list = ast.literal_eval(row[pred_col])
                pred_list = canonicalize_smiles(pred_list)

                if len(true_list) != len(pred_list):
                    continue

                if all(i == j for i, j in zip(true_list, pred_list)):
                    # Increment hits for top-k and all higher k
                    for j in range(k, top_n + 1):
                        top_n_hits[f'{pred_prefix}{j}'] += 1
                    break  # break after first match

        except Exception as e:
            if verbose:
                print(f"[Error] Row {idx}: {e}")

    # Convert counts to accuracy
    if total == 0:
        return {k: 0.0 for k in top_n_hits}

    top_n_accuracy = {
        k: round((v / total)*100, 4) for k, v in top_n_hits.items()
    }

    return top_n_accuracy


# Canonicalize the true SMILES
def canonicalize_smiles(smiles_list):
    if smiles_list is None:  # Handle NoneType gracefully
        return []
    canonical = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol:  # Check if MolFromSmiles succeeded
            canonical.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
        else:
            canonical.append(None)  # Mark invalid SMILES as None
    return canonical





#device = 'cuda:3'

def list_int_to_mech(list_intermediates):
    return '>>'.join(list_intermediates)

def our_canonicalizer(smiles):
    '''
    remove chirality information and canonicalize
    '''
    if smiles != '':
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''



def mass_checker(reactant_smile, product_smile):
    """
    Check if the total atom count (including hydrogens) of a product matches the reactant.

    Parameters:
    - reactant_smile: SMILES string of the reactant.
    - product_smile: SMILES string of the product.

    Returns:
    - True if atom counts match, False otherwise.
    """
    reactant_mol = Chem.AddHs(Chem.MolFromSmiles(reactant_smile))
    product_mol = Chem.AddHs(Chem.MolFromSmiles(product_smile))

    return reactant_mol.GetNumAtoms() == product_mol.GetNumAtoms()


def charge_len_checker(product_smile):
    """
    Calculate the number of charged atoms in a product.

    Parameters:
    - product_smile: SMILES string of the product.

    Returns:
    - Integer count of the charged atoms.
    """
    product_mol = Chem.MolFromSmiles(product_smile)
    charges = [atom.GetFormalCharge() for atom in product_mol.GetAtoms() if atom.GetFormalCharge() != 0]
    return len(charges)


def neg_atom_priority_diff(pdt_smiles_list):
    """
    Compare negative charges only within the unique parts of SMILES strings, ignoring common substructures.

    Parameters:
    - pdt_smiles_list: List of SMILES strings to compare.

    Returns:
    - The preferred SMILES string based on negative charge prioritization.
    """
    from rdkit import Chem

    # Split SMILES strings into sets of components
    split_smiles = [set(smiles.split('.')) for smiles in pdt_smiles_list]
    
    # Find common components across all SMILES strings
    common_components = set.intersection(*split_smiles)

    # Remove common components from each SMILES string's set
    unique_smiles = [components - common_components for components in split_smiles]

    # Handle only the unique parts for comparison
    unique_dict = {}  # Map original SMILES to its unique components as RDKit mols
    for idx, unique_parts in enumerate(unique_smiles):
        unique_mols = []
        for part in unique_parts:
            mol = Chem.MolFromSmiles(part)
            if mol:
                unique_mols.append((part, mol))
        unique_dict[pdt_smiles_list[idx]] = unique_mols

    # Evaluate negative charges in the unique parts
    neg_priority = []
    for original_smiles, unique_mols in unique_dict.items():
        max_priority = 0
        for part, mol in unique_mols:
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() < 0:
                    symbol = atom.GetSymbol()
                    if symbol == 'O':
                        max_priority = max(max_priority, 3)
                    elif symbol == 'N':
                        max_priority = max(max_priority, 2)
                    elif symbol == 'C':
                        max_priority = max(max_priority, 1)
        neg_priority.append((original_smiles, max_priority))
    
    # Sort by highest negative charge priority
    neg_priority.sort(key=lambda x: x[1], reverse=True)

    # Return the SMILES string with the highest negative charge priority
    return neg_priority[0][0] if neg_priority else pdt_smiles_list[0]


def process_results(results_dict):
    """
    Process the results dictionary to handle ties based on mass, charge checking,
    and prioritization of negative charges within unique product components.

    Parameters:
    - results_dict: Dictionary containing reactants and ranked products.

    Returns:
    - Modified results_dict with adjusted scores for non-best products where applicable.
    """


    #------------new addition, even before same score grouping, mass imbalance penalize---
    for key, value in results_dict.items():
        if key.startswith('Top') and isinstance(value, dict):
            product = value['product']
            #print('product:', product)
            score = value['score']
            if not mass_checker(results_dict['Reactants'], product):
                #print('I am here:', results_dict['Reactants'], product)
                value['score'] = score*0.1
    #-----------------------------------------------------------------------

    reactant_smile = results_dict['Reactants']
    score_groups = {}

    # Group products by score up to 5 decimal places
    for key, value in results_dict.items():
        if key.startswith('Top') and isinstance(value, dict):
            score = value['score'] #round(value['score'], 5)
            product = value['product']
            pred_actions = [i.split('_')[0] for i in value['pred_actions']]
            pred_actions = ''.join(pred_actions)
            score_groups.setdefault(score, []).append([product, pred_actions])

    updated_results_dict = {k: v.copy() if isinstance(v, dict) else v for k, v in results_dict.items()}

    for score, pdt_list_with_actions in score_groups.items():
        pdt_smiles_list = [p[0] for p in pdt_list_with_actions]
        actions_list = [p[1] for p in pdt_list_with_actions]

        if len(pdt_smiles_list) > 1:
            # Check if all matches are identical
            charge_len_data = []
            all_match = True

            for pdt_smile in pdt_smiles_list:
                charge_len = charge_len_checker(pdt_smile)
                mass_match = mass_checker(reactant_smile, pdt_smile)
                charge_len_data.append((pdt_smile, charge_len, mass_match))

                # Check if any mass mismatch or charge length difference exists
                if not mass_match or charge_len != charge_len_data[0][1]:
                    all_match = False
            
            #print('charge_len_data:', charge_len_data)

            # Skip scoring adjustments if all_match and no meaningful distinctions exist
            if all_match:
                # Check if further prioritization (e.g., for negative charges) is needed
                #if all(action == 'CCR' for action in actions_list):
                if all((action == 'CCR' or action == 'RR') for action in actions_list):
                    preferred_smiles = neg_atom_priority_diff(pdt_smiles_list)
                    for key, value in updated_results_dict.items():
                        if key.startswith('Top') and isinstance(value, dict):
                            product = value['product']
                            if product in pdt_smiles_list:
                                if product == preferred_smiles:
                                    value['score'] = score
                                else:
                                    value['score'] = score * 0.1  # Penalize non-preferred
                continue  # Skip further processing for this group

            # Apply scoring logic when distinctions exist
            sorted_products = sorted(
                charge_len_data,
                key=lambda x: (x[1], not x[2]),  # Sort by charge_len, then by mass_match
                reverse=False
            )
            best_product = sorted_products[0][0]

            for key, value in updated_results_dict.items():
                if key.startswith('Top') and isinstance(value, dict) and value['product'] in [entry[0] for entry in charge_len_data]:
                    if value['product'] == best_product:
                        value['score'] = score
                    else:
                        value['score'] = score * 0.1

    return updated_results_dict




