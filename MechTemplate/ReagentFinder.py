



#from atom mapped rxn get the reagents

from rdkit import Chem



def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags

def bond_to_smarts(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])
    bond_smarts = bond.GetSmarts()
    if bond_smarts == '':
        bond_smarts = '-'

    return '{}{}{}'.format(atoms[0], bond_smarts, atoms[1])



def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True # must be true for atom mapping

    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if REMOTE:
        if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
        # may be wrong information due to wrong atom mapping
        if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True

    # add or break bonds
    if atom_neighbors(atom1) != atom_neighbors(atom2): return True

    # change bonds
    bonds1 = sorted([bond_to_smarts(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_smarts(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2: return True

    return False

def atom_neighbors(atom):
    neighbor = []
    for n in atom.GetNeighbors():
        neighbor.append(n.GetAtomMapNum())
    return sorted(neighbor)


def get_changed_atoms(reactants, products):
    '''Looks at mapped atoms in a reaction and determines which ones changed'''

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)
    #print('prod_atom_tags:', prod_atom_tags)

    if VERBOSE: print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if VERBOSE: print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    #print('reac_atom_tags:', reac_atom_tags)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if VERBOSE: print('warning: different atom tags appear in reactants and products')
        #err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if VERBOSE: print('warning: total number of tagged atoms differ, stoichometry != 1?')
        #err = 1

    # Find differences
    changed_atoms = [] # actual reactant atom species
    changed_atom_tags = [] # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):
        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atom_tags: # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    #[clear_isotope(reactant) for reactant in reactants]
    #[clear_isotope(product) for product in products]


    if VERBOSE:
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))

    return changed_atoms, changed_atom_tags, err


def get_reagent_smiles(reactants, products):
    """
    Identifies reagents in a reaction (unchanged reactant molecules)
    and returns their SMILES strings.

    Parameters:
        reactants (list): List of RDKit molecule objects (reactants).
        products (list): List of RDKit molecule objects (products).

    Returns:
        reagent_smiles (list): SMILES strings of reagent molecules.
    """
    reagent_smiles = []

    # Extract tagged atoms and tags from reactants and products
    reac_atoms, reac_atom_tags = [], []
    prod_atoms, prod_atom_tags = [], []

    for reactant in reactants:
        atoms, tags = get_tagged_atoms_from_mol(reactant)
        reac_atoms.extend(atoms)
        reac_atom_tags.extend(tags)

    for product in products:
        atoms, tags = get_tagged_atoms_from_mol(product)
        prod_atoms.extend(atoms)
        prod_atom_tags.extend(tags)

    # Initialize a list to track whether reactant molecules are unchanged
    reac_molecule_status = [True] * len(reactants)

    print(reac_atom_tags, prod_atom_tags)
    # Iterate through reactant molecules
    for mol_idx, reactant in enumerate(reactants):
        reac_mol_atoms = [atom for atom in reactant.GetAtoms() if atom.HasProp('molAtomMapNumber')]

        for atom in reac_mol_atoms:
            reac_tag = atom.GetProp('molAtomMapNumber')

            # Check corresponding product atoms
            found_match = False
            for i, prod_tag in enumerate(prod_atom_tags):
                if reac_tag == prod_tag:  # Matching atom map number
                    if atoms_are_different(atom, prod_atoms[i]):  # Check if atom changes
                        reac_molecule_status[mol_idx] = False  # Molecule changes
                        found_match = True
                        break

            if not found_match and reac_tag not in prod_atom_tags:
                # If reactant atom is missing in the products
                reac_molecule_status[mol_idx] = False

        # If the molecule remains unchanged, add its SMILES
        if reac_molecule_status[mol_idx]:
            reagent_smiles.append(Chem.MolToSmiles(reactant))

    return reagent_smiles

rxn_smiles = '[Br:1][C:2]([Br:3])([Br:4])[Br:34].[CH3:5][C:6](=[O:7])[O:8][CH2:9][c:10]1[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]1[N:17]([C@H:18]([CH3:19])[CH2:20][CH2:21][CH2:22][OH:23])[S:24](=[O:25])(=[O:26])[c:27]1[cH:28][cH:29][c:30]([Cl:31])[cH:32][cH:33]1.[cH:39]1[cH:40][cH:41][c:36]([P:35]([c:42]2[cH:43][cH:44][cH:45][cH:46][cH:47]2)[c:48]2[cH:53][cH:52][cH:51][cH:50][cH:49]2)[cH:37][cH:38]1>>[Br:1][C-:2]([Br:3])[Br:4].[CH3:5][C:6](=[O:7])[O:8][CH2:9][c:10]1[cH:11][cH:12][c:13]([Cl:14])[cH:15][c:16]1[N:17]([C@H:18]([CH3:19])[CH2:20][CH2:21][CH2:22][OH:23])[S:24](=[O:25])(=[O:26])[c:27]1[cH:28][cH:29][c:30]([Cl:31])[cH:32][cH:33]1.[Br:34][P+:35]([c:36]1[cH:37][cH:38][cH:39][cH:40][cH:41]1)([c:42]1[cH:43][cH:44][cH:45][cH:46][cH:47]1)[c:48]1[cH:49][cH:50][cH:51][cH:52][cH:53]1'

reactants_con = rxn_smiles.split('>>')[0]
products_con = rxn_smiles.split('>>')[1]
reactants_list = reactants_con.split(".")
products_list = products_con.split('.')

reactants = [Chem.MolFromSmiles(i) for i in reactants_list]
products = [Chem.MolFromSmiles(i) for i in products_list]

VERBOSE = True
REMOTE = False
print(get_reagent_smiles(reactants, products))





