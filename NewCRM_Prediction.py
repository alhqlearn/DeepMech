


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

from my_inference_utils import  process_results, compute_top_n_accuracy
#----------------------------------Loading init_DeepMech-----------------------------

dataset = 'ID_Split42' # get the info of derived templates
scenario = 'Split42'
device = 'cuda' # cpu or cuda
model_name = 'DeepMech_%s' % scenario
model_path = 'models/%s.pth' % model_name
config_path = 'data/configs/default_config'
data_dir = 'data/%s' % dataset

args = {'data_dir': data_dir, 'model_path': model_path, 'config_path': config_path, 'device': device, 'mode': 'test'}
modelx, graph_functions, template_dicts, template_infos = init_DeepMech(args)

#-------------------------- Loading Classifier--------------------------------------
from reactivity_classifier_AttentiveFP import predict_reactivity
print('classifier loaded')
#-------------------------------------Functions--------------------------------

def beam_search_last_element2(start, expand_fn, beam_width=2, max_steps=10, alpha=1.0):
    """
    Beam search with length normalization to balance sequence length.
    Args:
        start (str): Starting sequence.
        expand_fn (callable): Function to expand a given word into next-word probabilities.
        eos_classifier (callable): Function to predict whether a word is the end of a sequence.
        beam_width (int): Number of candidates to keep at each step.
        max_steps (int): Maximum number of steps to run the search.
        alpha (float): Length normalization factor. Typically 0.6 <= alpha <= 1.0.
    Returns:
        list: Sorted list of (normalized_probability, sequence).
    """
    candidates = [(0.0, start)]  # (log_prob, sequence)
    finished = []  # Store completed sequences

    for step in range(max_steps):
        new_candidates = []
        all_finished = True  # Assume all sequences are finished initially

        for log_prob, seq in candidates:
            last_word = seq.split()[-1]  # Extract the last word
            #is_eos = eos_classifier([last_word])  # Check if sequence should end
            is_eos = np.array(predict_reactivity(last_word))
            #print('is_eos ', is_eos)
            #print('is_eos type', type(is_eos))

            if not is_eos:
                # Sequence is finished, apply length normalization
                length_norm = (len(seq.split()) ** alpha)
                normalized_score = log_prob / length_norm
                finished.append((normalized_score, seq))
            else:
                # Sequence is not finished; expand it
                all_finished = False
                outputs = expand_fn(last_word)
                
                for new_word, new_prob in outputs:
                    # Update sequence and log probability
                    full_seq = seq + " " + new_word
                    new_log_prob = log_prob + math.log(new_prob)
                    new_candidates.append((new_log_prob, full_seq))

        if all_finished:
            # Stop if all sequences are finished
            break
        
        # Keep the top beam_width candidates for the next step
        candidates = heapq.nlargest(beam_width, new_candidates, key=lambda x: x[0])
        
    # Combine finished sequences and any remaining candidates
    length_normalized = [
        (log_prob / (len(seq.split()) ** alpha), seq)
        for log_prob, seq in candidates
    ]
    finished.extend(length_normalized)

    # Return sorted finished sequences based on normalized probabilities
    return sorted(finished, key=lambda x: x[0], reverse=True)


def model_predict(reactants):
    sep = False
    verbose = False
    results_df, results_dict = predict_product(args, reactants, modelx, graph_functions,
                                           template_dicts, template_infos,
                                           verbose = verbose, sep = sep, collect_n = 200)

    #print('before:', results_dict)
    #adding the heuristic here:
    results_dict = process_results(results_dict)
    #print('after:', results_dict)
    products = []
    scores = []
    for key, value in results_dict.items():
        if key[:3] == 'Top':
            only_product = results_dict[key]['product']
            products.append(only_product)
            scores.append(results_dict[key]['score'])
    output = [(i, j) for i, j in zip(products[:5], scores[:5])]
    return output

#-----------------------------------------------------------------------------

start_sequence = "Cc1c(Cl)cccc1.CN(c2ccc([P+]([C@]34C[C@H]5C[C@@H](C4)C[C@@H](C3)C5)([C@]67C[C@H]8C[C@@H](C7)C[C@@H](C6)C8)[Pd])cc2)C.CC(C)([O-])C.CCCCN.[Na+]"
results = beam_search_last_element2(start_sequence, model_predict, beam_width=2, max_steps=10, alpha=0.5)
x = results[0][1].split(' ')
print(results)






