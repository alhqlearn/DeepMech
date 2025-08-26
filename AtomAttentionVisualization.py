import os, sys, re, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing

from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions, AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import PowerNorm

from dgllife.utils import (
    WeaveAtomFeaturizer, CanonicalBondFeaturizer, 
    mol_to_bigraph, smiles_to_bigraph
)

sys.path.append('./scripts/')
from models import DeepMech
from scripts.get_edit import combined_edit
from MechTemplate.template_collector import Collector


# Disable RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# ================== Data Loading ==================
template_file_path = "./data/ID_Split42"

def load_templates(template_file_path):
    template_dicts = {}
    for site in ['real', 'virtual']:
        template_df = pd.read_csv(f"{template_file_path}/{site}_templates.csv")
        template_dict = {
            template_df['Class'][i]: template_df['Template'][i].split('_') 
            for i in template_df.index
        }
        print(f"loaded {len(template_dict)} {site} templates")
        template_dicts[site[0]] = template_dict

    template_infos = pd.read_csv(f"{template_file_path}/template_infos.csv")
    template_infos = {
        template_infos['Template'][i]: {
            'edit_site': eval(template_infos['edit_site'][i]),
            'change_H': eval(template_infos['change_H'][i]),
            'change_C': eval(template_infos['change_C'][i]),
            'change_S': eval(template_infos['change_S'][i])
        } for i in template_infos.index
    }
    return template_dicts, template_infos


# ================== Graph Utils ==================
def get_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        for bond in atom.GetBonds():
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    V = [(a, b) for a in A for b in A if a != b and (a, b) not in B]
    return np.array(V), np.array(B)


def get_adm(mol, max_distance=6):
    mol_size = mol.GetNumAtoms()
    distance_matrix = np.ones((mol_size, mol_size)) * (max_distance + 1)
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1   # remote (different molecule)
    dm[dm > max_distance] = max_distance
    dm[dm == -1] = max_distance + 1
    distance_matrix[:dm.shape[0], :dm.shape[1]] = dm
    return distance_matrix


def pad_atom_distance_matrix(adm_list):
    max_size = max(adm.shape[0] for adm in adm_list)
    adm_list = [
        torch.tensor(
            np.pad(adm, (0, max_size - adm.shape[0]), 'maximum')
        ).unsqueeze(0).long() 
        for adm in adm_list
    ]
    return torch.cat(adm_list, dim=0)


def predict(model, bg, adms, bonds_dicts):
    adms = adms.to(device)
    bg = bg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    edge_feats = bg.edata.pop('e').to(device)
    return model(bg, adms, bonds_dicts, node_feats, edge_feats)


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


# ================== Load Templates ==================
template_dicts, template_infos = load_templates(template_file_path)

smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
atom_types = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 
    'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs'
]
node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
graph_function = lambda s: smiles_to_graph(
    s, node_featurizer=node_featurizer, 
    edge_featurizer=edge_featurizer, 
    canonical_atom_order=False
)


# ================== Model Setup ==================
Template_rn = len(pd.read_csv(f"{template_file_path}/real_templates.csv"))
Template_vn = len(pd.read_csv(f"{template_file_path}/virtual_templates.csv"))

model = DeepMech(
    node_in_feats=80,
    edge_in_feats=13,
    node_out_feats=256,
    edge_hidden_feats=32,
    num_step_message_passing=3,
    attention_heads=8,
    attention_layers=3,
    Template_rn=Template_rn,
    Template_vn=Template_vn
)

model.load_state_dict(
    torch.load("models/DeepMech_Split42.pth", map_location='cpu')['model_state_dict']
)
device = 'cpu'
model = model.to(device)


# ================== Example Run ================== 
smiles = 'COC(CCCO[P+](c1ccccc1)(c1ccccc1)c1ccccc1)OC.[Br-].BrC(Br)Br'

#-------------------------------------------------------------------------------------
dglgraph = graph_function(smiles)
mol_with_atom_index(Chem.MolFromSmiles(smiles))

adms = pad_atom_distance_matrix([get_adm(Chem.MolFromSmiles(smiles))])
v_bonds, r_bonds = get_bonds(smiles)
bonds_dicts = {
    'virtual': [torch.from_numpy(v_bonds).long()],
    'real': [torch.from_numpy(r_bonds).long()]
}

with torch.no_grad():
    pred_VT, pred_RT, rout_dict_v, rout_dict_r, pred_VI, pred_RI, attentions = predict(
        model, dglgraph, adms, bonds_dicts
    )

pred_v = torch.nn.Softmax(dim=1)(pred_VT)
pred_r = torch.nn.Softmax(dim=1)(pred_RT)
pred_vi, pred_ri = pred_VI[0].cpu(), pred_RI[0].cpu()

collect_n = 100
pred_types, pred_sites, pred_scores = combined_edit(
    v_bonds, r_bonds, pred_vi, pred_ri, pred_v, pred_r, collect_n
)

reactant, reagents, sep = smiles, '', 'mix'
collector = Collector(reactant, template_infos, reagents, sep, verbose=1)


# ================== Attention Visualization ==================
atom_attention, bond_attention = attentions[0], attentions[1]
atom_att = atom_attention[0].squeeze(0)  # first layer
num_heads = atom_att.shape[0]

fig, axes = plt.subplots(1, num_heads, figsize=(num_heads * 2.5, 3))
for i in range(num_heads):
    ax = axes[i] if num_heads > 1 else axes
    im = ax.imshow(atom_att[i].detach().cpu().numpy(), cmap='viridis')
    ax.set_title(f'Head {i}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# Average attention
att_avg = atom_att.mean(dim=0)
att_norm = (att_avg - att_avg.min()) / (att_avg.max() + 1e-8)

plt.figure(figsize=(4, 4))
im = plt.imshow(att_norm.detach().cpu().numpy(), cmap='viridis')
plt.colorbar(im)
plt.title('Average Normalized Attention')
plt.axis('off')
plt.tight_layout()
plt.show()

# Attention per atom
attn_avg = atom_att.mean(dim=0)
attn_per_atom = attn_avg.mean(dim=0)
attn_norm = (attn_per_atom - attn_per_atom.min()) / (attn_per_atom.max() - attn_per_atom.min())

# Color atoms
mol = Chem.MolFromSmiles(smiles)
mol_with_atom_index(mol)
atom_weights = attn_norm
cmap = cm.get_cmap('Reds')
plt_colors = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap)
atom_colors = {i: plt_colors.to_rgba(atom_weights[i].item()) for i in range(dglgraph.number_of_nodes())}

rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(mol)

drawer = rdMolDraw2D.MolDraw2DSVG(600, 600)
drawer.SetFontSize(5)
drawer.drawOptions().updateAtomPalette({7: (0.5, 0.5, 1)})
mol = rdMolDraw2D.PrepareMolForDrawing(mol)

drawer.DrawMolecule(mol, highlightAtoms=range(dglgraph.number_of_nodes()), highlightBonds=[], highlightAtomColors=atom_colors)
drawer.FinishDrawing()
svg = drawer.GetDrawingText().replace('svg:', '')
display(SVG(svg))


# --- NEW: Save as PNG ---
png_drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
png_drawer.SetFontSize(5)
png_drawer.drawOptions().updateAtomPalette({7: (0.5, 0.5, 1)})
png_drawer.DrawMolecule(mol, highlightAtoms=range(dglgraph.number_of_nodes()),
                        highlightBonds=[], highlightAtomColors=atom_colors)
png_drawer.FinishDrawing()

with open("attention_highlight.png", "wb") as f:
    f.write(png_drawer.GetDrawingText())

