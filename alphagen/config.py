from dataclasses import dataclass
import yaml
import torch

### Configs
@dataclass
class ModelConfig:
    d_emb:   int = 384
    d_model: int = 384
    n_head:  int = 16
    d_inner: int = 384
    n_layer: int = 6

@dataclass
class DatasetConfig:
    dataset_prefix: str = 'MonoamineR_human_0_aug_no_Q96RJ0_P21917'
    dataset_dir:    str = '/media/andrius/Extreme SSD/datasets'
    max_smiles_len: int = 100

@dataclass
class TrainerConfig:
    epochs:               int = 100
    batch_size:           int = 32
    eval_every_n_batches: int = 500
    save_every_n_batches: int = 100
    lr:                   float = 2e-4

@dataclass
class RunnerConfig:
    """ Main configuration for the runner """
    model:   ModelConfig   = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    trainer: TrainerConfig = TrainerConfig()
    model_dir: str = 'trained_models'
    model_num: int = 0
    dev = torch.device('cuda')
    devices = [0]


def load_config_from_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    config = RunnerConfig(
        model=ModelConfig(**config_dict['model']),
        dataset=DatasetConfig(**config_dict['dataset']),
        trainer=TrainerConfig(**config_dict['trainer']))

    return config

def save_config(runner):
    config_dict = dict(
        model = runner.config.model.__dict__,
        dataset = runner.config.dataset.__dict__,
        trainer = runner.config.trainer.__dict__)
    
    with open(f'{runner.config.model_dir}/config.yaml', 'w') as f:
        yaml.dump(config_dict, f)
