

import os, subprocess
from meeko import MoleculePreparation
from rdkit import Chem

class Docking:
    def __init__(self) -> None:
        self.docker = list
        self.adfr_suite_path = ''
        self.ligand_preparator = MoleculePreparation()


    def prepare_ligand(self, smiles: str, output_dir: str, output_prefix: str) -> str:
        """ Prepare ligand for docking, return ligand pdbqt path"""
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            protonated_ligand = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(protonated_ligand)
            self.ligand_preparator.prepare(protonated_ligand)
            
            # Write to pdbqt
            ligand_pdbqt_path = os.path.join(output_dir, f"{output_prefix}.pdbqt")
            self.ligand_preparator.write_pdbqt_file(ligand_pdbqt_path)

        except Exception as e:
            ligand_pdbqt_path = ''
            print(e)

        return ligand_pdbqt_path


    def prepare_target(self, pdb_path: str, output_dir: str, chain='A', output_prefix=None):
        """ Prepare target for docking, return target pdbqt path"""

        ## If the target consists of more than a single chain, we need to split it into separate files
        chain_pdb_path = pdb_path.replace('.pdb', '_%s.pdb' % chain)
        cmd = f'pdb_selchain -{chain} {pdb_path} | pdb_delhetatm | pdb_tidy > {chain_pdb_path}'
        run_executable(cmd)

        ## Prepare the target for docking
        target_pdbqt_path = pdb_path.replace('.pdb', '.pdbqt')
        prepare_receptor_binary = os.path.join(self.adfr_suite_path, 'bin', 'prepare_receptor')
        cmd = f'{prepare_receptor_binary} -r {chain_pdb_path} -o {target_pdbqt_path} -A checkhydrogens'
        run_executable(cmd)

        return target_pdbqt_path


    def build_docking_config(self, target_pdbqt, ligand_pdbqt, bbox_size=(20,20,20), bbox_loc=(0,0,0)):
        """ Build docking config file, return config file path"""
        
        config_path = ""
        return config_path


    def dock(self, output_dir, config=None, target_pdbqt=None, 
             ligand_pdbqt=None, bbox_size=(20,20,20), bbox_loc=(0,0,0), device="gpu"):
        
        """ Docking, return docking result path 
            If config file is provided as an arg, use it to do docking"""

        assert (config or (target_pdbqt and ligand_pdbqt))
        
        if config:
            pass
        else:
            config = self.build_docking_config(target_pdbqt, ligand_pdbqt, bbox_size=bbox_size)

        if device == "gpu":
            result_path = self.run_vina_gpu(config)
        else:
            raise ValueError("Device not supported")

        return result_path


    def run_vina_gpu(self, config):
        """ Run vina docking on gpu, return docking result path """

        cmd = "vina --config %s" % config
        output = run_executable(cmd)

        result_path = ""
        return result_path


    def process_stdout(self, stdout):
        """ Processes the stdout of Vina, returns the affinity of each docking orientation. """

        result = ""
        return result

        

def run_executable(cmd):
    """ Run executable command and return from stdout """

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout.decode("utf-8")
    

if __name__ == "__main__":
    pass