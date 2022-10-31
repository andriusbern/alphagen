import os
import shutil
import torch
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import VocSmiles
from alphagen.utils.SmilesEnumerator import SmilesEnumerator

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

def generate_alternative_smiles(smiles, enumerator, max_alternatives=10, n_attempts=100):

    alternatives, sdict = [], {}
    for i in range(n_attempts):

        new_smile = enumerator.randomize_smiles(smiles)
        if sdict.get(new_smile) is None:
            sdict[new_smile] = 1
            alternatives.append(new_smile)

    return alternatives[:max_alternatives]


# def get_all_counts(dataset_path):
#     """
#     Returns a dictionary of counts for each protein in the dataset
#     """
#     protein_id_idx = 9
#     pchembl_val_idx = 21
#     smiles_idx = 4
#     with open(dataset_path, 'r') as data:
#         dataset = data.readlines()
    
#     count_dict = {}
#     with open(dataset_path, 'r') as papyrus:
#         print('Processing papyrus...')
#         header = papyrus.readline()
#         for _ in tqdm(range(LENGTH_PAPYRUS)): # Process all papyrus entries
#             try:
#                 entry = papyrus.readline()
#                 attributes = entry.split('\t')
#                 pid = attributes[protein_id_idx]
#                 if count_dict.get(pid) is None:
#                     count_dict[pid] = 1
#                 else:
#                     count_dict[pid] += 1
#             except:
#                 pass
#     return count_dict


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

    print(processed_proteins)
    if protein_targets_file:
        with open(protein_targets_file, 'r') as prot_list:
            target_proteins = prot_list.readlines()
            target_proteins = [t.strip('\n').strip() for t in target_proteins]
    else:
        target_proteins = processed_proteins
   
    if process:
        clean_dataset(af_output_dir, output_dir, protein_ids=target_proteins)

    # Rescan
    # t_dir = output_dir
    # print(t_dir)
    # processed_proteins = [folder for folder in os.listdir(t_dir)
    #                       if os.path.isdir(os.path.join(t_dir, folder))]

    # processed_proteins = target_proteins
    print(processed_proteins)
    ## Process papyrus
    # Get the Ids of all proteins that were processed via AlphaFold
    voc = VocSmiles()
    words = set()

    target_proteins = list(set(target_proteins).intersection(set(processed_proteins)))
    print(target_proteins)

    # Create output file

    out_file = os.path.join(output_dir, output_dataset_prefix + '.tsv')
    with open(out_file, 'w') as f:
        f.write('PID\tpchembl\ttokens\n')

    protein_id_idx = 9
    pchembl_val_idx = 22
    smiles_idx = 4

    ## Smiles duplication
    sme = SmilesEnumerator()


    ### Most logical and fast way to do things would be the following:
    # 1. Read the target protein list
    # 2. Go through papyrus and collect all the smiles, pchembl values, store in a dict, count occurences


    # Processing line by line because of the dataset size
    with open(papyrus_file, 'r') as papyrus:
        print('Processing papyrus...')
        header = papyrus.readline()
        print([(i, attr) for i, attr in enumerate(header.split('\t'))])

        for i in tqdm(range(LENGTH_PAPYRUS)): # Process all papyrus entries
            try:

                entry = papyrus.readline()
                attributes = entry.split('\t')
                protein_id = attributes[protein_id_idx]
                pchembl_val = attributes[pchembl_val_idx].strip(';')

                if protein_id in target_proteins:

                    fl = float(pchembl_val)
                    # Check pchembl value
                    if fl < target_pchembl:
                        continue

                    smile = attributes[smiles_idx]

                    # Tokenize the smiles
                    tokenized = voc.split(smile)

                    if 5 < len(tokenized) <= 100:
                        words.update(tokenized)
                        
                        alternatives = generate_alternative_smiles(smile, sme, num_copies_of_smiles)
                        for smile in alternatives:
                            tokenized = voc.split(smile)
                            tokens = ' '.join(tokenized)

                            # If molecule seems legit, write it to the dataset
                            out = f'{protein_id}\t{pchembl_val}\t{tokens}\n'

                            with open(out_file, 'a') as f:
                                f.write(out)
            except:
                pass
                # print([(i, e) for i, e in enumerate(entry.split('\t'))])

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
    pchembl = 6.5
    af_output_dir = '/media/andrius/Extreme SSD/datasets/foldedPapyrus/proteins/'
    output_dataset_prefix = 'ar_kin_ccr_mono_slc6'
    output_dir = f'/home/andrius/git/datasets/{output_dataset_prefix}'
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


## The final big dataset:
# 1. ARs
# 2. Kinases
# 3. Monoamine receptors
# 4. CCRs
# 5. SLC6
