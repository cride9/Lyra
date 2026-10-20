from pathlib import Path
import torch
from .config import ModelConfig
from .data import Dataset
from .io import load_json, atomic_json
from .model import Lyra
from .runtime import device_for, gpu_session, load_checkpoint
from .tokenizer import Tokenizer
from .training import validate


def evaluate(checkpoint_path, dataset_path, output, count=128, split='test', device='auto',
             progress=print, cancelled=lambda: False):
    if split not in ('validation', 'test'):
        raise ValueError('Az értékelés validation vagy test részhalmazon fut.')
    saved = load_checkpoint(checkpoint_path)
    manifest = load_json(dataset_path)
    if manifest['hash'] != saved['dataset_hash']:
        raise ValueError('Az értékelési dataset eltér a modellétől.')
    device = device_for(device)
    config = ModelConfig(**saved['model_config'])
    dataset = Dataset(manifest, split, config)
    if cancelled():
        raise InterruptedError('Értékelés leállítva.')
    with gpu_session(device):
        torch.set_num_threads(2)
        model = Lyra(len(Tokenizer()), config).to(device)
        model.load_state_dict(saved['model'])
        result = dict(model='Lyra', split=split, checkpoint_step=saved['step'],
                      **validate(model, dataset, device, count))
    atomic_json(Path(output), result)
    progress(f'{split}: token-loss {result["token_loss"]:.4f}; {result["windows"]} különböző ablak.')
    return result
