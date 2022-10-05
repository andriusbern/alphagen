import torch
from torch import nn
from tqdm import tqdm

from alphagen.utils.dataset import ProteinSmilesDataset
from alphagen.model.smiles import AF2SmilesTransformer
from alphagen.utils.utils import VocSmiles, check_smiles
import os
from torch.utils.data import DataLoader
import time
import random
import numpy as np

from alphagen.config import DatasetConfig, RunnerConfig, load_config_from_file

## Suppress RDKit warnings
from rdkit import RDLogger
rdlog = RDLogger.logger()
rdlog.setLevel(RDLogger.CRITICAL)


## Voc and dataset
def load_voc(dataset_config: DatasetConfig):
    dataset_prefix = dataset_config.dataset_prefix
    dataset_dir = dataset_config.dataset_dir
    voc_file = os.path.join(dataset_dir, f'{dataset_prefix}', 'voc_smiles.txt')
    return VocSmiles(voc_file, max_len=dataset_config.max_smiles_len)

def load_dataset(voc, config: DatasetConfig):
    dataset_prefix = config.dataset_prefix
    dataset_dir = config.dataset_dir
    protein_list = os.path.join(dataset_dir, f'{dataset_prefix}', f'{dataset_prefix}.txt')
    dataset = ProteinSmilesDataset(dataset_dir, dataset_prefix=dataset_prefix, 
                                            voc=voc, protein_set=protein_list)
    return dataset


class Runner:
    def __init__(self, config: RunnerConfig=None, model_num: int=-1) -> None:
        
        if model_num > -1:
            model_dir = os.path.join('trained_models', str(model_num))
            config_path = os.path.join(model_dir, 'config.yaml')
            config = self.config = load_config_from_file(config_path)
            config.model_dir = model_dir

        self.config  = config
        self.voc     = load_voc(config.dataset)
        self.dataset = load_dataset(self.voc, config.dataset)
        self.model   = AF2SmilesTransformer(self.voc, **config.model.__dict__)

        if model_num > -1:
            weights_file = os.path.join(config.model_dir, 'model.pkg')
            self.model.load_state_dict(torch.load(weights_file, map_location=self.config.dev))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.trainer.lr)
        self.scaler = torch.cuda.amp.GradScaler()
        

    def save_model(self):
        """ Saves the model and config to a file """
        
        torch.save(self.model.state_dict(), os.path.join(self.config.model_dir, 'model.pkg'))
        save_config(self)


    def train(self, epochs: int=-1):
        """
        Training loop
        """
        dev = self.config.dev
        if epochs <= -1:
            epochs = self.config.trainer.epochs

        model_id = str(random.randint(0, 10000))
        model_dir = os.path.join(os.curdir, 'trained_models', model_id)
        self.config.model_dir = model_dir
        self.config.model_num = int(model_id)
        os.makedirs(model_dir)

        self.net = nn.DataParallel(self.model, device_ids=self.config.devices)
        dataloader = DataLoader(self.dataset, batch_size=self.config.trainer.batch_size, shuffle=True)
        
        t00 = time.time()
        batch_size = self.config.trainer.batch_size
        best = float('inf')
        running_loss = 5

        print(f'Model id: {model_id}, starting training...')
        for epoch in range(epochs):
            evaluation_n, n_samples_total = 0, 0

            print(f'\nEpoch {epoch + 1}/{epochs}\n')
            for i, src in enumerate(dataloader):
                proteins, smiles, pchembl = src
                proteins = proteins.to(dev)
                smiles = smiles.squeeze(1).to(dev)
                pchembl = pchembl.to(dev)
                self.optim.zero_grad()

                with torch.cuda.amp.autocast():
                    _, loss = self.net(x=smiles, mem=proteins, train=True)

                self.scaler.scale(loss.cuda()).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                running_loss = .98 * running_loss + .02 * loss.item() / batch_size
                n_samples_total += batch_size
                percent = n_samples_total/len(self.dataset)*100
                print('Batch {:4}, running loss: {:.4f} | {:8}/{:8} ({:.3f}%)'.format(
                      i, running_loss, n_samples_total, len(self.dataset), percent), end='\r')
                del loss
                
                # Saving model
                if i % self.config.trainer.save_every_n_batches == 0:
                    if running_loss < best:
                        self.save_model()
                        best = running_loss
                    
                ## Logging and evaluation
                if i % self.config.trainer.eval_every_n_batches == 0:
                    evaluation_n += 1
                    t = round((time.time() - t00)/3600, 3)
                    print(f'\nEvaluation: {evaluation_n}, t = {t}h  ||| {model_id}')
                    t0 = time.time()
                    
                    percentage, avg_len, valid = self.evaluate(molecules=50)
                    print(f'\nValid smiles: {percentage}% | Avg. length: {avg_len}')
            
                    with open(os.path.join(model_dir, 'training.log'), 'a') as log:
                        log.write(f'{evaluation_n} | {running_loss} | {t} | {percent} | {percentage} | {avg_len}\n')

                    with open(os.path.join(model_dir, 'smiles.log'), 'a') as smiles_log:
                        smiles = '\n'.join(valid)
                        smiles_log.write(f'Evaluation: {epoch}\n{smiles}\n{"-"*100}\n')


    def evaluate(self, molecules=50, verbose=False):
        """
        Evaluates the model by generating N smiles and checking if they are valid
        """
        dev = self.config.dev
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        tensor = torch.LongTensor([[self.voc.tk2ix['GO']]] * 1).to(dev).unsqueeze(1)
        x = tensor

        if self.net is None:
            self.net = torch.nn.DataParallel(self.model, device_ids=self.config.devices)

        generated_smiles = []
        lens = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print(f'Evaluating {i}/{molecules}', end='\r')
                protein_embedding, _, pchembl = data
                protein_embedding = protein_embedding.to(dev)
                pchembl = pchembl.to(dev)
                with torch.cuda.amp.autocast():
                    predictions, _ = self.net(x, mem=protein_embedding, train=False)

                smiles = predictions['tokens']
                if i >= molecules:
                    break
                
                smiles = smiles.squeeze().cpu().numpy()
                try:
                    ind = np.where(smiles == 0)[0][0]
                except:
                    ind = len(smiles)
                cut = smiles[:ind]
                decoded = self.voc.decode(cut, is_tk=False)
                generated_smiles += [decoded]
                lens += [len(decoded)]
            
        scores = check_smiles(generated_smiles)

        if verbose: 
            print("Valid generated smiles:")
        valid, n_valid = [], 0
        for i, smile in enumerate(generated_smiles):
            if scores[i] == 1:
                if smile != '' and smile != ' ':
                    valid += [smile]
                    n_valid += 1

        return n_valid/len(scores)*100, np.mean(lens), valid


    def targetted_generation(self, protein_id, fragments=None, repeat=1, batch_size=16):
        """
        Generates smiles for a target pid
        """
        dev = self.config.dev
        net = torch.nn.DataParallel(self.model, device_ids=self.config.devices)
        protein_embedding = self.dataset.protein_embeddings[protein_id]
        protein_embedding = protein_embedding.unsqueeze(0).repeat(batch_size, 1, 1).to(dev)

        print(f'Generating SMILES for {protein_id}')
        if fragments is None:
            x = torch.LongTensor([[self.voc.tk2ix['GO']]] * batch_size).to(dev)
        else:
            pass # Should add a way to construct smiles from fragments and pass them

        smiles_list = []
        with torch.no_grad():
            for i in range(repeat):
                predictions, _ = net(x, mem=protein_embedding, train=False)
                
                print(f'Generating SMILES {i+1}/{repeat}', end='\r')
                smiles = predictions['tokens'].cpu().numpy()
                for i in range(batch_size):
                    smile = self.voc.decode(smiles[i, :], is_tk=False)
                    smiles_list += [smile]
            
            scores = check_smiles(smiles_list)
            print(f'\n\nValidating: {scores.sum()}/{batch_size*repeat} of generated SMILES are valid.\n')
        
        return smiles_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_num', '-n', type=int, default=None)

    args = parser.parse_args()

    config = RunnerConfig()
    runner = Runner(config, model_num=args.model_num)

    runner.train()


