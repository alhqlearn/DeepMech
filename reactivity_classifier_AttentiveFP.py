import pandas as pd
import random
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer

import torch.nn as nn
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.model import AttentiveFPGNN
from dgllife.model import AttentiveFPReadout


import torch
import os
import random
import numpy as np
import ast

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import argparse
from rdkit import RDLogger 
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages


def extra_non_reactive_class(dff):
    all_labels_app = []
    for i in range(dff.shape[0]):
        string_list = dff['smiles'][i]
        spp =string_list.split(".")
        random.shuffle(spp)
        spp = spp.pop(0)
        #spp = ('.').join(spp)
        all_labels = [spp,[],7]
        all_labels_app.append(all_labels)
    all_labels_app = pd.DataFrame(all_labels_app, columns = dff.columns)
    return all_labels_app
    

# Function to retrieve values at specified positions
def get_values_at_positions(my_tuple, positions_list):
    return [my_tuple[pos] for pos in positions_list if pos < len(my_tuple)]
    
def atom_finder(smiles, ids):
    mol = Chem.MolFromSmiles(smiles)
    if len(ids) == 0:
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True)
        return shuffled_smiles, ids
    else: 
        atoms_interested = ast.literal_eval(ids)
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True) #'CC(C)=O.CC(C)(C)c1ccccc1' #here we just have to give shuffled or randomized smiles
        shuffled_mol = Chem.MolFromSmiles(shuffled_smiles)
        shuffled_ids  = shuffled_mol.GetSubstructMatch(mol)
        new_ids = get_values_at_positions(shuffled_ids, atoms_interested)
        return shuffled_smiles, new_ids

def smiles_augmentation(df): 
    information = []
    for i in range(df.shape[0]):
        react_random_list = [df[df.columns[0]][i], df[df.columns[1]][i]]
        information.append(react_random_list)    
    
    return information
        
def concat_feature_reactive_atom(graph_feat, changed_atoms):
    smiles_list = []
    target_list = []
    for i in range(len(changed_atoms)):
        smiles_list.append(changed_atoms[i][0])
        target_list.append(changed_atoms[i][1])
    # Convert the list of numbers to a list of tensors
    target_tensor_list = [torch.tensor([x], dtype=torch.float32) for x in target_list]
    return list(zip(smiles_list, graph_feat, target_tensor_list))
    
def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))


    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels
    
    
def Canon_SMILES_similarity(smiles_list):
    # Convert all SMILES strings to molecular objects
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Create an array of canonical smiles strings
    canonical_smiles_array = np.array([Chem.MolToSmiles(mol) if mol else None for mol in mol_list])

    # Use broadcasting to compare canonical smiles strings
    matrix = (canonical_smiles_array[:, None] != canonical_smiles_array).astype(np.float32)

    return torch.tensor(matrix)
    
    

class AttentiveFPPredictor_rxn(nn.Module):

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=1,
                 graph_feat_size=200,
                 n_tasks=8,
                 dropout=0.1):
        super(AttentiveFPPredictor_rxn, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks), nn.Sigmoid()
        )
        self.node_predict = nn.Sequential(nn.Linear(graph_feat_size, 1), nn.Sigmoid()
        )
    def forward(self, g, node_feats, edge_feats, get_node_weight=False):

        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            #node_feat_mod = self.node_predict(node_feats)
            return self.predict(g_feats), node_weights,  g_feats #node_feat_mod
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            #node_feat_mod = self.node_predict(node_feats)
            return self.predict(g_feats), g_feats # node_feat_mod
            
            
            
def weighted_binary_cross_entropy(output, target, weights=None):        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] *(target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    #print(loss)
    return torch.neg(torch.mean(loss))
    
    
def classifier_elementary_reactive(trained_model, sm, node_featurizer, edge_featurizer):
    smiles_graph = smiles_to_bigraph(sm, node_featurizer=node_featurizer,edge_featurizer=edge_featurizer, canonical_atom_order=False)
    n_feats_sm = smiles_graph.ndata.pop('hv').to(device)
    e_feats_sm = smiles_graph.edata.pop('he').to(device)
    pred_val, graph_feat = model(smiles_graph, n_feats_sm, e_feats_sm)
    pred_val_np = pred_val.detach().cpu().numpy()[0]
    #print('pred_val_np probability:', pred_val_np)
    threshold_class = 0.5
    pred_val_bin = [1 if pred_val_np >= threshold_class else 0]
    pred_val_bin
    return pred_val_bin
    
    
# Set device
device = "cpu"

# Initialize atom and bond featurizers
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')

# Get feature sizes
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

# Load trained model
#-------------oob trained model-----
#model_path = 'trained_classifier_reactive_or_nonreactive_10_07_25'  # Make sure this file exists
#model_path = 'trained_classifier_reactive_or_nonreactive_10_07_25_trial1'
#model_path = 'trained_classifier_wo_augm_10_07_25_WCE'
#model_path = 'trained_classifier_wo_augm_10_07_25_WCE_epoch20'

#-------------id trained model-----
model_path = 'ID_trained_classifier_wo_augm_10_07_25_WCE_epoch10' 
print('This model is used:', model_path)



model = AttentiveFPPredictor_rxn(
    node_feat_size=n_feats,
    edge_feat_size=e_feats,
    num_layers=2,
    num_timesteps=1,
    graph_feat_size=200,
    n_tasks=1,
    dropout=0.1
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_reactivity(smiles_string):
    """
    Predicts whether the given reaction SMILES is reactive (1) or non-reactive (0).
    
    Parameters:
        smiles_string (str): Reaction SMILES.
    
    Returns:
        int: 1 if reactive, 0 if non-reactive.
    """
    try:
        graph = smiles_to_bigraph(smiles_string, node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer, canonical_atom_order=False)
        n_feats_sm = graph.ndata.pop('hv').to(device)
        e_feats_sm = graph.edata.pop('he').to(device)

        with torch.no_grad():
            pred_val, _ = model(graph, n_feats_sm, e_feats_sm)
            pred_np = pred_val.detach().cpu().numpy()[0]
            #print('pred_np probability:', pred_np)
            return int(pred_np >= 0.5)
    except Exception as e:
        print(f"Error processing SMILES: {e}")
        return None

# Example usage (can be removed when importing as a module)
if __name__ == "__main__":
    smiles = 'CC(C)C1Nc2cccc3[nH]cc(c23)CC2COC(C)(C)N2C1=O.COc1cccc(OC)c1-c1ccccc1[P+]([Pd])(C1CCCCC1)C1CCCCC1.CC(C)(C)O.[Br-].[Na+]'
    prediction = predict_reactivity(smiles)
    print("Prediction (0 = non-reactive, 1 = reactive):", prediction)
