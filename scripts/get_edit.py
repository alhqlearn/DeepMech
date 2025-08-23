import os
import numpy as np
import pandas as pd

import time
import torch
import torch.nn as nn
import dgl

from utils import predict

def get_id_template(a, class_n):
    edit_idx = a//class_n
    template = a%class_n
    return (edit_idx, template)

def logit2pred(out, top_num):
    class_n = out.size(-1)
    #print('class_n:', class_n)
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    #for i in output_rank:
    #    edit_idx, template = get_id_template(i, class_n)
    #    if template != 0:
    #        print('I am here:', template)
    output_rank = [r for r in output_rank if get_id_template(r, class_n)[1] != 0][:top_num]
    #print('output_rankkkkkkkkkkkkkkkkk:', output_rank)
    selected_site = [get_id_template(a, class_n) for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
     
    return selected_site, selected_proba
    
def get_bond_site(graph, etype):
    atom_pair_list = torch.transpose(graph.adjacency_matrix(etype=etype).coalesce().indices(), 0, 1).numpy()
    atom_pair_list = [atom_pair_list_v[idx] for idx in pred_vi]


def combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, top_num):
    # Validate input sizes
    if not (len(virtual_bonds) >= max(pred_vi, default=0) and len(real_bonds) >= max(pred_ri, default=0)):
        raise ValueError("Invalid indices in pred_vi or pred_ri for given bonds.")

    pred_site_v, pred_score_v = logit2pred(pred_v, top_num)
    pred_site_r, pred_score_r = logit2pred(pred_r, top_num)
    #print(f"pred_site_v, pred_score_v, {pred_site_v},  {pred_score_v}")
    #print(f"pred_site_r, pred_score_r: {pred_site_r}, {pred_score_r}")
    # Safely pool bonds
    pooled_virtual_bonds = [virtual_bonds[idx] for idx in pred_vi if idx < len(virtual_bonds)]
    pooled_real_bonds = [real_bonds[idx] for idx in pred_ri if idx < len(real_bonds)]

    # Adjust pred_site lists
    try:
        pred_site_v = [(list(pooled_virtual_bonds[pred_site]), pred_temp)
                       for pred_site, pred_temp in pred_site_v if pred_site < len(pooled_virtual_bonds)]
        pred_site_r = [(list(pooled_real_bonds[pred_site]), pred_temp)
                       for pred_site, pred_temp in pred_site_r if pred_site < len(pooled_real_bonds)]
    except IndexError:
        raise ValueError("Mismatch between predicted sites and pooled bonds.")

    pred_sites = pred_site_v + pred_site_r
    pred_types = ['v'] * len(pred_site_v) + ['r'] * len(pred_site_r)
    pred_scores = pred_score_v + pred_score_r

    # Re-rank and filter
    if len(pred_scores) > top_num:
        pred_ranks = np.flip(np.argsort(pred_scores))[:top_num]
        pred_types = [pred_types[r] for r in pred_ranks]
        pred_sites = [pred_sites[r] for r in pred_ranks]
        pred_scores = [pred_scores[r] for r in pred_ranks]

    return pred_types, pred_sites, pred_scores


def get_bg_partition(bg, bonds_dicts):
    gs = dgl.unbatch(bg)
    v_sep = [0]
    r_sep = [0]
    for g, v_bonds, r_bonds in zip(gs, bonds_dicts['virtual'], bonds_dicts['real']):
        pooling_size = g.num_nodes()
        n_v_bonds = len(v_bonds)
        if n_v_bonds < pooling_size:
            v_sep.append(v_sep[-1] + n_v_bonds)
        else:
            v_sep.append(v_sep[-1] + pooling_size)
        n_r_bonds = len(r_bonds)
        if n_r_bonds < pooling_size:    
            r_sep.append(r_sep[-1] + n_r_bonds)
        else:
            r_sep.append(r_sep[-1] + pooling_size)
    return v_sep[1:], r_sep[1:]



def write_edits(args, model, test_loader):
    model.eval()
    with open(args['result_path'], 'w') as f:
        f.write('Test_id\tReactants\tReagents\t%s\n' % '\t'.join(['Prediction %s' % (i+1) for i in range(args['top_num'])]))
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                reactants, reagents, bg, adms, bonds_dicts = data
                #print('reactant, reagenst:', reactants, reagents)
                pred_VT, pred_RT, _, _, pred_VI, pred_RI, _  = predict(args, model, bg, adms, bonds_dicts)
                pred_VT = nn.Softmax(dim=1)(pred_VT)
                pred_RT = nn.Softmax(dim=1)(pred_RT)
                v_sep, r_sep = get_bg_partition(bg, bonds_dicts)
                start_v = 0
                start_r = 0
                print('\rWriting test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
                for i, (reactant, reagent) in enumerate(zip(reactants, reagents)):
                    end_v, end_r = v_sep[i], r_sep[i]
                    virtual_bonds, real_bonds = bonds_dicts['virtual'][i].numpy(), bonds_dicts['real'][i].numpy()
                    pred_vi, pred_ri = pred_VI[i].cpu(), pred_RI[i].cpu()
                    pred_v, pred_r = pred_VT[start_v:end_v], pred_RT[start_r:end_r]
                    #print('pred_v, pred_r=========>',pred_v.shape, pred_r.shape) 
                    
                    pred_types, pred_sites, pred_scores = combined_edit(virtual_bonds, real_bonds, pred_vi, pred_ri, pred_v, pred_r, args['top_num'])
                    #print('pred_types, pred_sites, pred_scores:--->', pred_types[:5], pred_sites[:5], pred_scores[:5])
                    test_id = (batch_id * args['batch_size']) + i
                    f.write('%s\t%s\t%s\t%s\n' % (test_id, reactant, reagent, '\t'.join(['(%s, %s, %s, %.3f)' % (pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i]) for i in range(args['top_num'])])))
                    start_v = end_v
                    start_r = end_r
    print ()
    return

