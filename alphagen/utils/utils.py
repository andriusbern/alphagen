import torch
import numpy as np
from rdkit.Chem.MolStandardize import rdMolStandardize
import re
from rdkit import Chem
import tqdm

"""
All of the functions here are taken from around DrugEx codebase. 
"""

def clean_mol(smile, is_deep=True):
    """Taken from dataset.py, modified to take/return a single smile"""

    smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
        .replace('[N]', 'N').replace('[B]', 'B') \
        .replace('[2H]', '[H]').replace('[3H]', '[H]')
    try:
        mol = Chem.MolFromSmiles(smile)
        if is_deep:
            mol = rdMolStandardize.ChargeParent(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        smile = Chem.CanonSmiles(smileR)
    except:
        print('Parsing Error:', smile)
        smile = None
    return smile


class VocSmiles:
    """A class for handling encoding/decoding from SMILES to an array of indices
        Taken from utils/vocab.py, slightly adjusted to fix bugs (token duplication)"""

    def __init__(self, init_from_file=None, max_len=100):
        self.control = ['_', 'GO']
        from_file = []
        if init_from_file:
            from_file = self.init_from_file(init_from_file)
            from_file = list(set(from_file))
            from_file.sort()
        self.words = self.control + from_file
        print(self.words)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len


    def encode(self, input):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        output = torch.zeros(len(input), self.max_len).long()
        for i, seq in enumerate(input):
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char]
        return output

    def decode(self, tensor, is_tk=True):
        """Takes an array of indices and returns the corresponding SMILES"""
        tokens = []
        for token in tensor:
            if not is_tk:
                token = self.ix2tk[int(token)]
            if token == 'EOS': break
            if token in self.control: continue
            tokens.append(token)
        smiles = "".join(tokens)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def split(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens + ['EOS']

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        words = []
        with open(file, 'r') as f:
            chars = f.read().split()
            words += sorted(set(chars))
        return words

    def calc_voc_fp(self, smiles, prefix=None):
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        for i, smile in enumerate(smiles):
            smile = clean_mol(smile)
            token = self.split(smile)
            if prefix is not None: token = [prefix] + token
            if len(token) > self.max_len: continue
            if {'C', 'c'}.isdisjoint(token): continue
            if not {'[Na]', '[Zn]'}.isdisjoint(token): continue
            fps[i, :] = self.encode(token)
        return fps
    

def standardize_mol(mol):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    carbon = Chem.MolFromSmarts('[#6]')
    salts = Chem.MolFromSmarts('[Na,Zn]')
    try:
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        mol = chooser.choose(mol)
        mol = charger.uncharge(mol)
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        # remove SMILES that do not contain carbon
        if len(mol.GetSubstructMatches(carbon)) == 0:
            return None
        # remove SMILES that still contain salts
        if len(mol.GetSubstructMatches(salts)) > 0:
            return None
        return Chem.CanonSmiles(smileR)
    except:
        print('Parsing Error:', Chem.MolToSmiles(mol))

    return None


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def check_smiles(smiles, frags=None):
    shape = (len(smiles), 1) if frags is None else (len(smiles), 2)
    valids = np.zeros(shape)
    for j, smile in enumerate(smiles):
        # 1. Check if SMILES can be parsed by rdkit
        try:
            mol = Chem.MolFromSmiles(smile)
            valids[j, 0] = 0 if mol is None else 1
        except:
            valids[j, 0] = 0
        if frags is not None:
            # 2. Check if SMILES contain given fragments
            try:
                subs = frags[j].split('.')
                subs = [Chem.MolFromSmiles(sub) for sub in subs]
                valids[j, 1] = np.all([mol.HasSubstructMatch(sub) for sub in subs])
            except:
                valids[j, 1] = 0
    return valids