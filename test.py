import torch
from tqdm import tqdm

from alphagen.utils.dataset import ProteinSmilesDataset
from alphagen.model.smiles import AF2SmilesTransformer
from alphagen.utils.utils import VocSmiles, check_smiles
from torch.utils.data import DataLoader

## Suppress RDKit warnings
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

dev = torch.device('cuda')
devices = [0]

from alphagen.trainer import Runner


if __name__ == "__main__":

    model_config = {}

    dataset_dir = '/media/andrius/Extreme SSD/datasets'
    trainer = Runner(model_num=360)

    protein_targets_file = '/home/andrius/git/datasets/MonoamineR_human_65.txt'
    with open(protein_targets_file, 'r') as prot_list:
        target_proteins = prot_list.readlines()
        target_proteins = [t.strip('\n').strip() for t in target_proteins]

    smiles = []
    smiles_dict = {}
    for protein in target_proteins:
        p_smiles = trainer.targetted_generation(protein_id=protein, batch_size=10, repeat=10)
        smiles_dict[protein] = p_smiles