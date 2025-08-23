

import time
import copy
import numpy as np
from functools import reduce
from collections import defaultdict
from itertools import permutations, combinations, islice

import rdkit
from rdkit import Chem, RDLogger
import signal
import networkx as nx

from .template_decoder import *

def demap(smiles):
    if smiles == None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]     
    return Chem.MolToSmiles(mol)

def combine_dict(d1, d2):
    for k, v in d1.items():
        if k in d2:
            if v != d2[k]:
                return False
        elif v in d2.values():
            return False
        else:
            d2[k] = v
    return d2
    

#def match_each(preds, trues, matched_idx, ):
#    '''
#    first match_each modification that we did
#    this resolved the halting problem for large permutations, but
#    couldn't solve the exact problem pertaining to aromatic system
#    '''
#    import random
#    num_samples = 500
#    #perms = list(permutations(preds, len(trues))) #for a large number of preds this stops, n!/(n-r)! permutation
#    all_perms = permutations(preds, len(trues))
#    perms = list(islice(all_perms, num_samples))
#    #perms = random.sample(all_perms, min(num_samples, len(all_perms))) #can't do with the generator
#    if len(preds) < len(trues):
#        return []
#
#    ts = [item for elem in trues for item in elem]
#    ms = []
#    for perm in perms:
#        ps = [item for elem in perm for item in elem]
#        m = {t:p for t, p in zip(ts, ps) if t != -1}
#        combined_dict = combine_dict(m, copy.copy(matched_idx))
#        if combined_dict:
#            ms.append(combined_dict)
#    return ms


def match_each(preds, trues, matched_idx):
    #print('preds:-->', preds, '\n')
    #print('trues:--->', trues, '\n')
    import random
    if len(trues) <= 6:
        #print('permutation for less than 6:', trues)
        #num_samples = 720
        #perms = list(permutations(preds, len(trues))) #for a large number of preds this stops, n!/(n-r)! permutation
        all_perms = permutations(preds, len(trues))
        perms = all_perms #list(islice(all_perms, num_samples))
        #perms = random.sample(all_perms, min(num_samples, len(all_perms))) #can't do with the generator
        if len(preds) < len(trues):
            return []

        ts = [item for elem in trues for item in elem]
        ms = []
        for perm in perms:
            ps = [item for elem in perm for item in elem]
            m = {t:p for t, p in zip(ts, ps) if t != -1}
            combined_dict = combine_dict(m, copy.copy(matched_idx))
            if combined_dict:
                ms.append(combined_dict)
    elif len(trues) > 6:
        print('subgraph is greater than 6:', trues)
        try:
            perms = graph_isomorphism_match(trues, preds)
            #print('permutation via graph_isomorphism:', perms)
            if len(preds) < len(trues):
                return []

            ts = [item for elem in trues for item in elem]
            ms = []
            for perm in perms:
                ps = [item for elem in perm for item in elem]
                m = {t:p for t, p in zip(ts, ps) if t != -1}
                combined_dict = combine_dict(m, copy.copy(matched_idx))
                if combined_dict:
                    ms.append(combined_dict)
        except Exception as e:
            print(f'An error occured: {e}')
            ms = []
            return ms
        

    return ms



#def match_each(preds, trues, matched_idx):
#    from itertools import permutations
#    import copy
#
#    ms = []  # Initialize empty match list
#
#    if len(trues) <= 6:
#        if len(preds) < len(trues):
#            return []
#
#        all_perms = permutations(preds, len(trues))
#        ts = [item for elem in trues for item in elem]
#
#        for perm in all_perms:
#            ps = [item for elem in perm for item in elem]
#            m = {t: p for t, p in zip(ts, ps) if t != -1}
#            combined_dict = combine_dict(m, copy.copy(matched_idx))
#            if combined_dict:
#                ms.append(combined_dict)
#
#    # If len(trues) > 6, do nothing (but function continues and returns empty list)
#    return ms



def graph_isomorphism_match(trues, preds):
    # Define graph G
    G = nx.Graph()
    G.add_edges_from(preds)
    
    # Define subgraph H
    H = nx.Graph()
    H.add_edges_from(trues)  # Subgraph resembling part of G
    
    # Step 3: Check for isomorphism
    gm = nx.isomorphism.GraphMatcher(G, H)
    
    # Step 4: Handle mapping
    mapped_graph_idx = []
    
    if gm.subgraph_is_isomorphic():
        #print("The graphs are isomorphic!")
        for i, mapping in enumerate(gm.subgraph_isomorphisms_iter(), start=1):
            #print(f"Match {i}: {mapping}")
            mapped_graph_idx.append(mapping)

    new_perms = []
    for i in mapped_graph_idx:
        reverse_i = {v: k for k, v in i.items()}
        x_mapped = [(reverse_i[a], reverse_i[b]) for a, b in trues]
        new_perms.append(x_mapped)
    
    return new_perms



'''
original match_each function

def match_each(preds, trues, matched_idx):
    #this is combination based matching
    if len(preds) < len(trues):
        return []
    
    ts = [item for elem in trues for item in elem]
    ms = []
    
    for comb in combinations(preds, len(trues)):
        ps = [item for elem in comb for item in elem]
        m = {t: p for t, p in zip(ts, ps) if t != -1}
        combined_dict = combine_dict(m, copy.copy(matched_idx))
        if combined_dict:
            ms.append(combined_dict)
    
    return ms
'''

def bidirect_len(bonds, return_len = True):
    bidirected = copy.copy(bonds)
    for bond in bonds:
        if (bond[1], bond[0]) not in bidirected:
            bidirected.append((bond[1], bond[0]))
    if return_len:
        return len(bidirected)
    else:
        return bidirected

def single_match(pred_action, pred_idx, template_actions):
    template_idxs = template_actions[pred_action]
    matches = []
    for temp_idx in template_idxs:
        matches.append({t:p for t, p in zip(temp_idx, pred_idx) if t != -1})
    return matches

def split_pred_idxs(pred_idxs):
    matched_idxs = []
    for vs in pred_idxs:
        matched_idxs += [v for v in vs if v not in matched_idxs]
    return matched_idxs
    
class Collector():
    def __init__(self, reactant, tempaltes_info, reagents, products = None, sep = False, verbose = False):
        self.templates_info = tempaltes_info
        self.reactant = reactant
        if str(reagents) == 'nan':
            self.reagents = ''
        else:
            self.reagents = reagents
        self.min_n_atoms = 1
        self.products = None
        self.non_reacts = []
        self.has_small_fragment = False
        self.verbose = verbose
        self.sep = sep
        
        self.predictions = defaultdict(dict)
        self.old_predictions = set()
        self.predicted_template = defaultdict(list)
        self.template_scores = defaultdict(dict)
        self.used_idx = defaultdict(list)
        self.predicted_roles = dict()

    #def clean_small_frags(self, products):
    #    if '[IH3]' in products:
    #        products = products.replace('[IH3]', '[IH]')
    #    return  '.'.join([product for product in products.split('.') if Chem.MolFromSmiles(product).GetNumAtoms() >= self.min_n_atoms])
        
        
    def clean_small_frags(self, products):
        # Replace '[IH3]' with '[IH]' in the products string
        if '[IH3]' in products:
            products = products.replace('[IH3]', '[IH]')
    
        # Filter products that meet the criteria
        valid_products = []
        for product in products.split('.'):
            mol = Chem.MolFromSmiles(product)
            if mol and mol.GetNumAtoms() >= self.min_n_atoms:
                valid_products.append(product)
        
        # Join the valid products with a dot and return
        return '.'.join(valid_products)

    
    def reconstruct_actions(self, template_roles, pred_idxs, recorded_actions):
        pred_actions = []
        for k, v in template_roles.items():
            for pred in v:
                if k != 'R':
                    pred_mech = '%s_%s_%s' % (k, pred_idxs[pred[0]], pred_idxs[pred[1]])
                    pred_actions.append('%s_%s_%s' % (k, pred_idxs[pred[0]], pred_idxs[pred[1]]))
                else:
                    pred_actions.append('%s_%s' % (k, pred_idxs[pred[0]]))
        return pred_actions


    def recursive_match(self, preds, trues, template_full, n_required_idx):
        matched_idxs = [{}]
        for edit_type in preds:
            if len(trues[edit_type]) == 0:
                continue
            if edit_type == 'R' and self.has_small_fragment and len(preds[edit_type]) == 0:
                continue
            if bidirect_len(preds[edit_type]) < bidirect_len(trues[edit_type]):
                return []
            new_matched_idxs = []
            for matched_idx in matched_idxs:
                #print('input to the matched_idx:\n', preds[edit_type], trues[edit_type], matched_idx, template_full, '\n')

                matched_idx = match_each(preds[edit_type], trues[edit_type], matched_idx)
                new_matched_idxs += matched_idx
            matched_idxs = new_matched_idxs
        matched_idxs = list(map(dict, set(tuple(sorted(d.items())) for d in matched_idxs if len(d) >= n_required_idx)))
        outputs = []
        for matched_id in matched_idxs:
            recorded_actions = self.template_scores[template_full]
            try:
                pred_actions = self.reconstruct_actions(trues, matched_id, recorded_actions)
                if sum([action not in recorded_actions for action in pred_actions]) > 0:
                    continue
                else:
                    outputs.append([matched_id, pred_actions])
            except Exception as e:
                if self.verbose:
                    print (e)
        return outputs

  

      

    def collect(self, template, H_code, C_code, S_code, pred_action, pred_idx, score):
        template_full = '%s_%s_%s_%s' % (template, H_code, C_code, S_code)
        #print('template_full:', template_full)
        template_info = self.templates_info[template_full]
        #print(f"template_info:\n{template_info}")
        conf_changes = {'H': template_info['change_H'], 'C': template_info['change_C'], 'S': template_info['change_S']}
        #print(f"conf_changes:\n{conf_changes}")
        template_actions = template_info['edit_site']
        #print(f"template_actions:\n{template_actions}")
        n_required_idx = len(set([atom for temp_action, bonds in template_actions.items() for bond in bonds for atom in bond if temp_action != 'R']))
        change_bond_only = len(template_actions['A']) + len(template_actions['B']) + len(template_actions['R']) == 0
        print(f"change_bond_only:\n{change_bond_only}")
        if change_bond_only:
            for i in pred_idx:
                self.template_scores[template_full][i] = score
        else:
            if pred_action != 'R':
                pred_mech = '%s_%s_%s' % (pred_action, pred_idx[0], pred_idx[1])
                pred_mech_inv = '%s_%s_%s' % (pred_action, pred_idx[1], pred_idx[0])
                print(f"pred_mech, pred_mech_inv:\n{pred_mech}, {pred_mech_inv}")
                if pred_mech not in self.template_scores[template_full]:
                    self.template_scores[template_full][pred_mech] = score
                    if pred_action == 'C':
                        self.template_scores[template_full][pred_mech_inv] = score
                        #print('\nI am looking here\n')
                        
            else:
                pred_mech = '%s_%s' % (pred_action, pred_idx[0])
                if pred_mech not in self.template_scores[template_full]:
                    self.template_scores[template_full][pred_mech] = score
            
        if n_required_idx >= 6 and change_bond_only:
            n_required_idx -= 2
            
        newly_pred_idxs = []
        if template_full not in self.predicted_template:
            #print('you are here:', 1)
            if change_bond_only:
                self.predicted_template[template_full] = [pred_idx]
                matched_idxs = split_pred_idxs(self.predicted_template[template_full])
                if len(matched_idxs) >= n_required_idx:
                    if self.verbose:
                        print ('pred_actions:', matched_idxs)
                        print ('template_actions:', template_actions)
                    self.predict(template_full, conf_changes, matched_idxs, template_actions, True)
            else:
                self.predicted_template[template_full] = [{edit_type:[] for edit_type in template_actions}]
                #print('see this:', self.predicted_template[template_full])
                self.predicted_template[template_full][0][pred_action].append(pred_idx)
                pred_idxs = self.predicted_template[template_full][0]
                #print(f"pred_idxs:\n{pred_idxs}")
                matched_idxs = self.recursive_match(pred_idxs, template_actions, template_full, n_required_idx)
                #print(f"matched_idxs:\n{matched_idxs}")
                for matched_idx in matched_idxs:
                    self.predict(template_full, conf_changes, matched_idx, template_actions)
                        
        elif change_bond_only:
            #print('you are here:', 2)
            self.predicted_template[template_full].append(pred_idx)
            matched_idxs = split_pred_idxs(self.predicted_template[template_full])
            if len(matched_idxs) >= n_required_idx:
                if self.verbose:
                    print ('pred_actions:', matched_idxs)
                    print ('template_actions:', template_actions)
                self.predict(template_full, conf_changes, matched_idxs, template_actions, True)
                
        else:
            #print('you are here:', 3)
            #print('inputs are:', template_actions, template_full, n_required_idx)
            for pred_idxs in self.predicted_template[template_full]: 
                #print('template colleted, pred_idxs example:', pred_idxs)
                pred_idxs[pred_action].append(pred_idx)
                #print(f'2nd pred_idxs:\n{pred_idxs}')
                start_time = time.time()
                try:
                    #print(f'input to the recursive_match:\n {pred_idxs}\n, {template_actions}\n, {template_full}\n, {n_required_idx}\n')
                    matched_idxs = self.recursive_match(pred_idxs, template_actions, template_full, n_required_idx)
                    #print('matched_idxs:-->\n', matched_idxs)
                except Exception as e:
                    if self.verbose:
                        print(f'error in recursive match:{e}')
                    continue #skip to the next iteration

                end_time = time.time()
                elapsed_time = end_time - start_time
                #print(f'Time taken for self.recursive_match: {elapsed_time} seconds')
                if self.verbose:
                    print ('pred_actions:', pred_idxs)
                    print ('template_actions:', template_actions)
                    
                for matched_idx in matched_idxs:
                    if self.verbose:
                        print ('matched_idx:', matched_idx)
                    self.predict(template_full, conf_changes, matched_idx, template_actions)
        return 
    
    def predict(self, template_full, conf_changes, pred_idxs, template_actions, change_bond_only = False):
        if not change_bond_only:
            pred_idxs, pred_actions = pred_idxs
            idx_code = ''.join([str(pred_idxs[k]) for k in pred_idxs.keys()])
            if idx_code in self.used_idx[template_full]:
                return False
            else:
                self.used_idx[template_full].append(idx_code)
        
        template, _, _, _ = template_full.split('_')
        #print('template:', template) 
        try:
            #print('template, conf_changes, pred_idxs, change_bond_only:',template, conf_changes, pred_idxs, change_bond_only,)
            matched_products, fit_temp = apply_template(self, template, conf_changes, pred_idxs, change_bond_only, self.verbose)
            #print('matched_products, fit_temp:', matched_products, fit_temp, '\n')
            if change_bond_only:
                pred_actions = []
                for matched_idx, matched_product in matched_products.items():
                    if matched_product in self.old_predictions:
                        continue
                    for idx in eval(matched_idx).values():
                        if 'C_%s' % idx not in pred_actions and idx in pred_idxs:
                            pred_actions.append('C_%s' % idx) 
            print('matched_products:', matched_products)
            if self.verbose:
                print ('matched_products:', matched_products)
                print (pred_actions)
        
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
                
        except Exception as e:
            #print('It is an error')
            if self.verbose:
                print (e)
            return 
        #print('self.old_predictions:', self.old_predictions) 
        newly_predicted = []
        
        if fit_temp == '([*:1])>>([*:1])': #I have added this if-else condition for no-rxn
            first_value = next(iter(matched_products.values()))
            for product in first_value.split('.'):
                newly_predicted.append(product)
        
        else:
            for matched_idx, products in matched_products.items():
                if self.products != '':
                    products = self.clean_small_frags(products)
                    #newly_predicted.append(products[0])
                '''
                I don't need this, so stopped, becasuse of this the products generated were not balanced to the reactants,
                this is desired as we made all of our templates atom-balanced

                for product in products.split('.'):
                    if product not in newly_predicted and product not in self.old_predictions:
                        newly_predicted.append(product)
                        self.old_predictions.add(product)
                '''
                #for product in products.split('.'):
                #    print('products-------->', products, '\n')
                #    if product not in newly_predicted:
                #        newly_predicted.append(product)

                #-------------change@161224@1842--------here
                #print('products-------->', products, '\n')
                #for product in products.split('.'):
                #    newly_predicted.append(product)         
                newly_predicted.append(products) 
                break #I am breaking here, not taking all the match products for a single template
                    
        if self.verbose:
            print ('predicted product(s):', newly_predicted)
         
        #print ('predicted product(s):', newly_predicted)
        #print('self.old_predictions:', self.old_predictions)
        if len(newly_predicted) != 0:
            if len(newly_predicted[0]) == 0:
                return 
            predicted_product = '.'.join(sorted(newly_predicted))
            #print('self.template_scores:--->\n', self.template_scores) 
            if change_bond_only:
                score = np.average([self.template_scores[template_full][int(action.split('_')[1])] for action in pred_actions])
            else:
                score = np.average([self.template_scores[template_full][action] for action in pred_actions])
            
            if predicted_product not in self.predictions:
                self.predictions[predicted_product] = {'template':fit_temp, 'pred_actions': pred_actions, 'pred_idx':pred_idxs, 'score':score}
        return
    
    
