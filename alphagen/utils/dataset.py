import os
import shutil
import pandas as pd
from torchvision.io import read_image
import torch
import pickle
import numpy as np
from tqdm import tqdm
from alphagen.utils.utils import VocSmiles
from torch.utils.data import Dataset as TorchDataset
from SmilesEnumerator import SmilesEnumerator
import random

LENGTH_PAPYRUS = 61085165


class ProteinDataset:
    """Class for loading and storing AlphaFold embeddings in torch.Tensor format"""

    def __init__(self, data_dir, voc, protein_set=None):
        self.data_dir = data_dir
        self.protein_dir = os.path.join(data_dir, 'foldedPapyrus', 'proteins')
        self.max_len = 768
        
        # Embedding dict accessible via protein_id
        self.protein_embeddings = self.load_protein_embeddings(protein_set)
        self.voc = voc
        
    def get_protein_embedding(self, protein_id, embedding_type='single'):
        """
        Returns a torch tensor of the target protein 
        """
        protein_dir = os.path.join(self.protein_dir, protein_id)
        embedding_file = os.path.join(protein_dir, f'{embedding_type}.npy')

        np_file = np.load(embedding_file)
        embedding = torch.from_numpy(np_file)
        n, f = embedding.shape
        output = torch.zeros(self.max_len, f)
        output[:n, :f] = embedding
        return output, embedding.shape

    def load_protein_embeddings(self, protein_set, embedding_type='single', scale=True):

        protein_dict = {}
        protein_ids =  [folder for folder in os.listdir(self.protein_dir) 
                        if os.path.isdir(os.path.join(self.protein_dir, folder))]
        
        if protein_set is not None:
            with open(protein_set, 'r') as pids:
                pids = pids.readlines()
                pids = [p.strip('\n') for p in pids]
            protein_ids = list(set(pids).intersection(set(protein_ids)))


        # Scaling in the range [-1, 1] (based on min/max values of each feature)
        embs = [self.get_protein_embedding(pid)[0] for pid in protein_ids]
        concat = torch.cat(embs, dim=0)
        emb_max = torch.max(concat, dim=0)[0]
        emb_min = torch.min(concat, dim=0)[0]

        for i, pid in enumerate(protein_ids):
            scaled_embedding = (embs[i] - emb_min) / (emb_max - emb_min) * 2 - 1
            protein_dict[pid] = scaled_embedding

        return protein_dict

        
class ProteinSmilesDataset(TorchDataset, ProteinDataset):
    """Class for storing both AF and papyrus data"""

    def __init__(self, data_dir, dataset_prefix, batch_size=32, **kwargs):
        
        super(ProteinSmilesDataset, self).__init__(data_dir=data_dir, **kwargs)
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, f'{dataset_prefix}', f'{dataset_prefix}.tsv')
        self.tsv_dataset = self.read_dataset()
        print(f'Dataset: len: {len(self)}')

    def __len__(self):
        return len(self.tsv_dataset)

    def __getitem__(self, idx):
        """
        Returns torch tensors for:
            1) Protein embeddings
            2) Standardized SMILES
            3) Pchembl value scalar
        """

        data = self.tsv_dataset[idx]
        pid, pchembl, tokens = data.split('\t')
        protein_embeddings = self.protein_embeddings[pid]
        try:
            encoded_smiles = self.voc.encode([tokens.strip('\n').split(' ')])
        except:
            data = self.tsv_dataset[0]
            pid, pchembl, tokens = data.split('\t')
            encoded_smiles = self.voc.encode([tokens.strip('\n').split(' ')])
        try:
            # pchembl = torch.tensor(float(pchembl))
            pchembl = torch.tensor(float(0.0))
        except:
            pass

        return protein_embeddings, encoded_smiles, pchembl

    def read_dataset(self):
        """
        Reads the dataset and returns a list of strings with the following format
            protein_id \t SMILES \t pchembl
        """
        with open(self.dataset_path, 'r') as data:
            dataset = data.readlines()
        return dataset[1:]
    
    def write_dataset(self, dataset_path):
        """
        Writes the dataset to a file
        """
        with open(dataset_path, 'w') as data:
            for line in self.tsv_dataset:
                data.write(line)

    def augment(self, min_ratio=0.5):
        """
        Performs data augmentation - this should really be done at the creation state, because most of this code
        is merely redoing whatever process was used to create the dataset in the first place

        The procedure is as follows - since each target in the dataset has a varying number of SMILES, we
        try to equalize this number by randomly permuting existing SMILES to reach the desired minimum number
        
        The min_ratio is the minimum number of SMILES per target that we want to reach (if max was)
        """
        ## Count the number of smiles per by iterating through the dataset
        count_dict, smiles_dict = {}, {}
        for data in self.tsv_dataset:
            pid, pchembl, tokens = data.split('\t')
            if count_dict.get(pid) is None:
                count_dict[pid] = 1
            else:
                count_dict[pid] += 1

            if smiles_dict.get(pid) is None:
                smiles_dict[pid] = []
            else:
                smiles_dict[pid] += 1
        
        pids, counts = count_dict.items()
        max_count = max(counts.values())

        ## Augment the dataset by generating alternative smiles for targets
        # that have a low number of datapoints
        for protein, count in counts.items():
            if count < max_count * min_ratio:
                smiles = self.protein_embeddings[protein]
                for i in range(max_count - count):
                    self.tsv_dataset.append(f'{protein}\t{smiles[i]}\t0.0')


        
        
        
def generate_alternatives(smiles, num_alternatives):
    """
    Generates alternative SMILES for given SMILES strings
    """
    enumerator = SmilesEnumerator()

    augmented = []
    for i in range(num_alternatives):
        smile = random.sample(smiles, 1)
        alternative = enumerator.randomize_smiles(smiles)
        augmented.append(alternative)
    
    return augmented
        

            


def clean_dataset(af_output_dir, output_dir):
    """
    Cleaning
    """
    protein_ids = [folder for folder in os.listdir(af_output_dir) \
                   if os.path.isdir(os.path.join(af_output_dir, folder))]
    print(protein_ids)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'proteins'), exist_ok=True)

    # Process AlphaFold's output
    for protein in protein_ids:
        
        src_dir = os.path.join(af_output_dir, protein)
        protein_dir = os.path.join(output_dir, 'proteins', protein)
        os.makedirs(protein_dir, exist_ok=True)

        path = os.path.join(src_dir, 'result_model_1_pred_0.pkl')
        pkl = open(path, 'rb')
        result = pickle.load(pkl)
        reps = result['representations']
        single, struct = reps['single'], reps['structure_module']

        np.save(os.path.join(protein_dir, 'single.npy'), single)
        np.save(os.path.join(protein_dir, 'struct.npy'), struct)

        files_to_copy = ['ranked_0.pdb',
                         'unrelaxed_model_1_pred_0.pdb', 
                         f'{protein}.fasta']

        for file in files_to_copy:
            try:
                shutil.copy(os.path.join(src_dir, file), os.path.join(protein_dir, file))
            except:
                pass


def get_all_counts(dataset_path):
    """
    Returns a dictionary of counts for each protein in the dataset
    """
    protein_id_idx = 9
    pchembl_val_idx = 21
    smiles_idx = 4
    with open(dataset_path, 'r') as data:
        dataset = data.readlines()
    
    count_dict = {}
    with open(dataset_path, 'r') as papyrus:
        print('Processing papyrus...')
        header = papyrus.readline()
        for _ in tqdm(range(LENGTH_PAPYRUS)): # Process all papyrus entries
            try:
                entry = papyrus.readline()
                attributes = entry.split('\t')
                pid = attributes[protein_id_idx]
                if count_dict.get(pid) is None:
                    count_dict[pid] = 1
                else:
                    count_dict[pid] += 1
            except:
                pass
    return count_dict


def build_dataset(af_output_dir, output_dir, papyrus_file, output_dataset_prefix, 
                  save_voc=True, process=False):
    """
    For building a compact dataset with only really required data
    1. List all processed proteins
    2. Process the output if needed, move to target dir
    3. Read papyrus line by line

        - if protein in entry is processed, store this as a line
        - at the end each protein has a list of smiles
        - they can be processed via the usual pipeline
    4. Conjoin all the data as
        pid  smiles  pX

    """

    if process:
        clean_dataset(af_output_dir, output_dir)

    ## Process papyrus
    # Get the Ids of all proteins that were processed via AlphaFold
    protein_ids = [folder for folder in os.listdir(af_output_dir) 
                   if os.path.isdir(os.path.join(af_output_dir, folder))]
    voc = VocSmiles()
    words = set()
    
    # Create output file
    out_file = os.path.join(output_dir, output_dataset_prefix + '.tsv')
    with open(out_file, 'w') as f:
        f.write('PID\tpchembl\ttokens\n')

    protein_id_idx = 9
    pchembl_val_idx = 21
    smiles_idx = 4

    with open(papyrus_file, 'r') as papyrus:
        print('Processing papyrus...')
        header = papyrus.readline()
        print([(i, attr) for i, attr in enumerate(header.split('\t'))])

        for _ in tqdm(range(LENGTH_PAPYRUS)): # Process all papyrus entries
            try:
                entry = papyrus.readline()
                attributes = entry.split('\t')

                # If protein_id was processed by alphafold add an entry to new dataset
                protein_id = attributes[protein_id_idx]
                if protein_id in protein_ids: 
                    pchembl_val = attributes[pchembl_val_idx][:5]
                    smile = attributes[smiles_idx]

                    # Tokenize the smiles
                    tokenized = voc.split(smile)
                    if 10 < len(tokenized) <= 100: 
                        words.update(tokenized)
                        tokens = ' '.join(tokenized)

                        # If molecule seems legit, write it to the dataset
                        out = f'{protein_id}\t{pchembl_val}\t{tokens}\n'
                        open(out_file, 'a').write(out).close()
            except:
                print('Something went wrong with this entry:')

    # Save vocabulary file
    if save_voc:
        print('Saving vocabulary...')
        with open(os.path.join(output_dir, 'voc_smiles.txt'), 'w') as voc:
            voc.write('\n'.join(sorted(words)))


if __name__ == "__main__":
    af_output_dir = '/home/andrius/data/500_proteins/proteins'
    output_dir = '/home/andrius/data/500_proteins'
    papyrus_file = '/media/andrius/Extreme SSD/05.4_combined_set_with_stereochemistry.tsv'
    output_dataset_prefix = 'dataset'

    build_dataset(af_output_dir, output_dir, papyrus_file, output_dataset_prefix, process=False)

