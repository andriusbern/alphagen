import os
import sys
import time
import shutil
import pickle
import numpy as np

from batch_process import FASTA_DIR

MSA_DIR = '/scratch/bernataviciusa/af_data/output/msas'
DATASET_DIR = '/data/bernataviciusa/af_data/final'
FASTA_DIR = '/scratch/bernataviciusa/af_data/papyrus/fasta'
    

def gpu_processing_done(directory):
    """
    Checks for AlphaFold output files at the directory based on protein_id

    Need a try, except for creating, deleting the gpu_running flag in case it breaks
    """
    if os.path.exists(directory):
        files_to_check = ['result_model_1_pred_0.pkl',
                          'ranked_0.pdb',
                          'relaxed_model_1_pred_0.pdb']

        exists = [os.path.isfile(os.path.join(directory, f)) for f in files_to_check]

    return all(exists)


def flag(directory, flag):
    """
    Creates an empty flag file to indicate that other machine 
    is already running the MSA/GPU stage to avoid overlaps and 
    deadlocks
    """
    open(os.path.join(directory, f'{flag}'), 'w').close()


def msa_done(protein_id):
    """
    After the MSA stage completes, move output to /data accessible to 
    GPU machines
    """

    # Copy fasta
    fasta_path = os.path.join(FASTA_DIR, f'{protein_id}.fasta')
    shutil.copy(fasta_path, os.path.join(MSA_DIR, protein_id, f'{protein_id}.fasta'))

    # Copy whole subfolder
    source = os.path.join(MSA_DIR, protein_id)
    destination = os.path.join(DATASET_DIR)
    shutil.copytree(source, destination)


## These should be run at the very end
def clean_alphafold_output(protein_id):
    """
    Sorts the .pkl and .pdb files; extracts the relevant internal representations and 
    stores them as .npy files
    """
    try:
        prot_dir = os.path.join(DATASET_DIR, protein_id)

        process_result_pkl(prot_dir)

        files = os.listdir(prot_dir)
        pdb_dir = os.path.join(prot_dir, 'pdb')
        pkl_dir = os.path.join(prot_dir, 'pkl')
        os.mkdir(pdb_dir)
        os.mkdir(pkl_dir)

        for output_file in files:
            if output_file.endswith('.pdb'):
                shutil.move(os.path.join(prot_dir, output_file), pdb_dir)
            elif output_file.endswith('.pkl'):
                shutil.move(os.path.join(prot_dir, output_file), pkl_dir)

    except:
        print(f'Could not clean up AF output for protein {protein_id}.')


def process_result_pkl(directory, n_models=1):
    """
    Store evoformer and structure module representations in protein_id/representations 
    folder
    """
    for num in range(0, n_models):
        
        path = os.path.join(directory, 'result_model_{num+1}_pred_0.pkl')
        pkl = open(path, 'rb')
        result = pickle.load(pkl)

        reps = result['representations']
        single, struct = reps['single'], reps['structure_module']

        rep_dir = os.path.join(directory, 'representations')
        os.mkdir(rep_dir)
        
        suffix = f'_{num+1}.npy' if n_models == 1 else '.npy'

        np.save(os.path.join(rep_dir, f'single{suffix}'), single)
        np.save(os.path.join(rep_dir, f'struct{suffix}'), struct)

