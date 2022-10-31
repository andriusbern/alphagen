import os
import shutil
import torch
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from alphagen.utils.utils import VocSmiles
from alphagen.utils.SmilesEnumerator import SmilesEnumerator
import random

LENGTH_PAPYRUS = 61085165

class ProteinDataset:
    """Class for loading and storing AlphaFold embeddings in torch.Tensor format"""

    def __init__(self, data_dir, voc, protein_set=None) -> None:
        self.data_dir = data_dir
        self.protein_dir = data_dir
        self.max_len = 450
        self.protein_dir = os.path.join(data_dir, 'foldedPapyrus', 'proteins')
        self.max_len = 768
       
        # Embedding dict accessible via protein_id
        self.protein_embeddings = self.load_protein_embeddings(protein_set)
        self.voc = voc
       
    def get_protein_embedding(self, protein_id, embedding_type='single'):
        """
        Returns a torch tensor of the target protein
        """
        protein_dir = os.path.join(self.data_dir, protein_id)
        protein_dir = os.path.join(self.protein_dir, protein_id)
        embedding_file = os.path.join(protein_dir, f'{embedding_type}.npy')

        np_file = np.load(embedding_file)
        embedding = torch.from_numpy(np_file)
        n, f = embedding.shape
        output = torch.zeros(self.max_len, f)
        output[:n, :f] = embedding
        return output, n

    def load_protein_embeddings(self, protein_set, embedding_type='single'):

        protein_dict = {}
        protein_ids =  [folder for folder in os.listdir(self.protein_dir)
                        if os.path.isdir(os.path.join(self.protein_dir, folder))]
       
        if protein_set is not None:
            with open(protein_set, 'r') as pids:
                pids = pids.readlines()
                pids = [p.strip('\n') for p in pids]
            protein_ids = list(set(pids).intersection(set(protein_ids)))

        for pid in protein_ids:
            protein_dict[pid] = self.get_protein_embedding(pid)[0]
           
        return protein_dict

       
class ProteinSmilesDataset(torch.utils.data.Dataset, ProteinDataset):
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

        # def get_line(idx):
        # while True:
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
        Reads the dataset, makes sure that embeddings for each protein are available
        """
       
        filtered = []
        with open(self.dataset_path, 'r') as data:
            for line in data.readlines():
               
                pid, pchembl, tokens = line.split('\t')
                if self.protein_embeddings.get(pid) is None:
                    continue
                filtered.append(line)
       
        return filtered 


def clean_dataset(af_output_dir, output_dir, protein_ids=None):
    """
    Cleaning
    """

    if protein_ids is None:

        protein_ids = [folder for folder in os.listdir(af_output_dir) \
                       if os.path.isdir(os.path.join(af_output_dir, folder))]

    print(protein_ids)
    print(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    # Process AlphaFold's output
    for i, protein in enumerate(protein_ids):
        try:
            print(f'Processing {protein} | {i}/{len(protein_ids)}', end='\r')
           
            src_dir = os.path.join(af_output_dir, protein)
            protein_dir = os.path.join(output_dir, protein)

            path = os.path.join(src_dir, 'result_model_1_pred_0.pkl')
            pkl = open(path, 'rb')
            result = pickle.load(pkl)
            reps = result['representations']
            single, struct = reps['single'], reps['structure_module']

            os.makedirs(protein_dir, exist_ok=True)
            np.save(os.path.join(protein_dir, 'single.npy'), single)
            np.save(os.path.join(protein_dir, 'struct.npy'), struct)

            files_to_copy = ['ranked_0.pdb',
                            'unrelaxed_model_1_pred_0.pdb',
                            f'{protein}.fasta']

            for file in files_to_copy:
               
                shutil.copy(os.path.join(src_dir, file), os.path.join(protein_dir, file))
        except:
            pass

def generate_alternative_smiles(smiles, n_alternatives=10, n_attempts=50):
    enumerator = SmilesEnumerator()
    alternatives, sdict = [], {}
    for i in range(n_attempts):

        new_smile = enumerator.randomize_smiles(smiles)
        if sdict.get(new_smile) is None:
            sdict[new_smile] = 1
            alternatives.append(new_smile)
        
        if len(alternatives) == n_alternatives:
            break

    return alternatives


class Ligand:
    def __init__(self, string, target, pchembl) -> None:
        self.string = string
        self.length = len(string)
        self.target = target
        self.pchembl = pchembl

    def __len__(self) -> int:
        return len(self.string)

    def __repr__(self) -> str:
        return f'{self.target} L={len(self)} | {self.pchembl}'


def build_dataset(af_output_dir, output_dir, papyrus_file, output_dataset_prefix,
                  protein_targets_file=None, save_voc=True, process=False, target_pchembl=0.0,
                  num_copies_of_smiles=1):
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
    processed_proteins = [folder for folder in os.listdir(af_output_dir)
                   if os.path.isdir(os.path.join(af_output_dir, folder))]

    if protein_targets_file:
        with open(protein_targets_file, 'r') as prot_list:
            target_proteins = prot_list.readlines()
            target_proteins = [t.strip('\n').strip() for t in target_proteins]
    else:
        target_proteins = processed_proteins
   
    if process:
        clean_dataset(af_output_dir, output_dir, protein_ids=target_proteins)

    target_proteins = list(set(target_proteins).intersection(set(processed_proteins)))
    print(target_proteins)

    ##################
    ### Read papyrus
    protein_id_idx = 9
    pchembl_val_idx = 22
    smiles_idx = 4
    smiles_dict = {pid: [] for pid in target_proteins}
    with open(papyrus_file, 'r') as papyrus:
        print('Processing papyrus...')
        header = papyrus.readline()
        print([(i, attr) for i, attr in enumerate(header.split('\t'))])

        for i in tqdm(range(LENGTH_PAPYRUS)): # Process all papyrus entries
            try:

                entry = papyrus.readline()
                attributes = entry.split('\t')
                # print(attributes)
                protein_id = attributes[protein_id_idx].strip('_WT')
                # print(protein_id)
                pchembl_val = attributes[pchembl_val_idx].strip(';')
                # print(pchembl_val)
                if protein_id in target_proteins:
                    try:
                        fl = float(pchembl_val)
                    except:
                        fl = 1
                        # print(attributes)
                    # Check pchembl value
                    if fl < target_pchembl:
                        continue

                    smile_str = attributes[smiles_idx]
                    smile = Ligand(smile_str, protein_id, fl)
                    smiles_dict[protein_id] += [smile]
            except:
                # print(protein_id, fl, smile_str)
                pass

    # Counting the number of smiles per target
    max_num_smiles = max([len(s) for s in smiles_dict.values()])
    ratio = 0.1
    min_number_of_smiles = round(max_num_smiles * ratio)

    #####################
    ## Augmentation
    for pid, smiles in smiles_dict.items():
        n_smiles = len(smiles)
        # if n_smiles < min_number_of_smiles:
        num_augmentations = (min_number_of_smiles - n_smiles)
        print(n_smiles, num_augmentations)
        if num_augmentations > 0 and n_smiles > 0:
            num_augmentations_per_mol = num_augmentations // n_smiles + 1
            ## Generate N alternatives for each smile
            alternative_smiles = []
            for smile in smiles:
                alt_smiles = generate_alternative_smiles(smile.string, n_alternatives=num_augmentations_per_mol)
                alternative_smiles += [Ligand(s, smile.target, smile.pchembl) for s in alt_smiles]
            shuffled_indices = np.random.permutation(len(alternative_smiles))[:num_augmentations]
            smiles_dict[pid] += [alternative_smiles[i] for i in shuffled_indices]
    # print(smiles_dict[pid])

    ##################
    # Create output file
    voc = VocSmiles()
    words = set()
    out_file = os.path.join(output_dir, output_dataset_prefix + '.tsv')
    with open(out_file, 'w') as f:
        f.write('PID\tpchembl\ttokens\n')
        for smiles in smiles_dict.values():
            print(len(smiles))
            # smiles = smiles_dict[pid]
            for smile in smiles:

                tokenized = voc.split(smile.string)
                if 5 < len(tokenized) <= 100:
                    words.update(tokenized)
                    tokens = ' '.join(tokenized)
                    f.write(f'{smile.target}\t{smile.pchembl}\t{tokens}\n')
                
    # Save vocabulary file
    if save_voc:
        print('Saving vocabulary...')
        with open(os.path.join(output_dir, 'voc_smiles.txt'), 'w') as voc:
            voc.write('\n'.join(sorted(words)))


def create_protein_subset_file(output_dir, write_out_file=False):
    """
    Parses the filtered papyrus files and creates a text file with all the unique
    protein accession numbers contained within it
    """
    target_id = 0

    directory = '/Users/amac/git/datasets'
    prefix = 'results_filtered_'
    names = ['ARs', 'ARs_high', 'Kin_human_high', 'CCRs_human_high', 'MonoamineR_human_high',
            'MonoamineR_human', 'SLC6_high']

    filenames = [prefix + suffix +  '.txt' for suffix in names]
    paths = [os.path.join(directory, f) for f in filenames]
    target = paths[target_id]
    out_file = os.path.join(output_dir, names[target_id] + '.txt')

    data = pd.read_csv(target, sep='\t')
    proteins = data['accession'].unique()

    # For writing out the output file that contains the filtered protein IDs
    if write_out_file:

        with open(out_file, 'w') as out:
            for entry in proteins:
                if '.' not in entry:
                    out.write(f'{entry}\n')


if __name__ == "__main__":
    pchembl = 0
    af_output_dir = '/media/andrius/Extreme SSD/datasets/foldedPapyrus/proteins/'
    output_dataset_prefix = 'ar_kin_ccr_mono_slc6'
    output_dir = f'/home/andrius/git/datasets/{output_dataset_prefix}'
    # papyrus_file = '/home/andrius/Downloads/results_filtered_MonoamineR_human.txt'
    papyrus_file = '/media/andrius/Extreme SSD/05.4_combined_set_with_stereochemistry.tsv'

    # af_output_dir = '/data/bernataviciusa/af_data/final'
    # output_dir = '/data/bernataviciusa/af_data/data/kinases'
    # papyrus_file = '/data/bernataviciusa/af_data/papyrus/papyrus_.tsv'
    # output_dataset_prefix = 'dataset'
    protein_targets_file = '/home/andrius/git/datasets/ar_kin_ccr_mono.txt'

    build_dataset(af_output_dir, output_dir, papyrus_file,
                  output_dataset_prefix, protein_targets_file=protein_targets_file,
                process=False, target_pchembl=pchembl,
                num_copies_of_smiles=10)

    # create_subset('/Users/amac/git/datasets', write_out_file=True)

    # build_dataset()
   

                    # # Molecules in papyrus seem to be already standardized, skip
                    # mol = Chem.MolFromSmiles(smile)
                    # standardized = standardize_mol(mol)

## Experimental setup:
# 1. Try the ARs set - 26K entries + 24 targets, see if learning happens, can also try out somewhat smaller nets
#       See if the model can start overfitting
# 2. Monoamine human - 3.3M entries, 36 targets
# 3. Kin human high  - 264k entries, 435 targets
# 4. Use the accesions in Kin_human_high to assemble another one with all entries (should be quite a lot)