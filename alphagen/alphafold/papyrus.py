## Script for fetching all required fasta files based on protein IDs in papyrus

from urllib import request
from tqdm import tqdm
import os, sys


def download_fasta(protein_id, output_folder):
    """
    Retrieves the fasta file from Uniprot based on protein ID
    """
    try:
        remote_url = f'https://uniprot.org/uniprot/{protein_id}.fasta'
        local_file = f'{output_folder}/{protein_id}.fasta'
        open(local_file, 'a').close()
        request.urlretrieve(remote_url, local_file)
    except:
        print(f'Fasta file for {protein_id} could not be downloaded.')


def read_fasta(file_path):
    """
    Reads a .fasta file and returns a Protein Sequence object
    """
    try:

        with open(file_path, 'r') as fasta:
            content = fasta.readlines()
            header = content[0]
            sequence = content[1:]
            stripped = [line.strip('\n') for line in sequence]
            sequence = ''.join(stripped)
            protein_id = header.split('|')[1].strip()
        
        protein_seq = ProteinSequence(sequence=sequence, protein_id=protein_id, info=header)
        return protein_seq

    except:
        return None


class ProteinSequence:
    """
    Simple object for storing protein info
    """
    def __init__(self, sequence=None, protein_id=None, filename=None, info=None):
        self.info = info
        self.seq = sequence
        self.protein_id = protein_id
        self.len = None
        self.ligand_idxs = [] # Datapoint line idxs in papyrus
        self.n_ligands = 0  # Number of datapoints in papyrus
    
    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        return f"Protein Seq: {self.protein_id}, length: \
                {len(self)}, ligands: {self.n_ligands}"


def load_protein_sequence(protein_id, data_folder):
    """
    If needed download the .fasta file based on protein ID and return a 
    ProteinSequence object
    """

    # Check if downloaded
    filename = f'{data_folder}/{protein_id}/{protein_id}.fasta'
    if not os.path.isfile(filename):
        download_fasta(protein_id, data_folder)
        
    protein_seq = read_fasta(filename)
    return protein_seq


def get_papyrus_proteins(papyrus_file, output_folder, start=0, end=61085165):
    """
    Scans through the whole papyrus dataset, and downloads all required .fasta files from 
    uniprot, based on protein accesion IDs (attribute[9] in papyrus)

    Returns a dict of ProteinSequence objects indexed via protein_ids
    """
    if not os.path.isfile(papyrus_file):
        print('Papyrus file not found.')
        sys.exit()

    end = 61085165 if not end else end
    
    with open(papyrus_file, 'r') as papyrus:

        header = papyrus.readline()
        proteins = {}

        for idx in tqdm(range(start, end)):
            entry = papyrus.readline()
            attributes = entry.split('\t')
            protein_id = attributes[9]

            if not proteins.get(protein_id):
                p_sequence = load_protein_sequence(protein_id, data_folder=output_folder)
                proteins[protein_id] = p_sequence
            else:
                proteins[protein_id].n_ligands += 1
                # if store_idx:
                #     proteins[protein_id].ligand_idxs.append(idx)

    return proteins
