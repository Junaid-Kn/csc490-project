import logging
import torch
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig  # Hydra uses DictConfig for configs

# Allow PyTorch to safely load DictConfig
torch.serialization.add_safe_globals([DictConfig])

torch.serialization.add_safe_globals([ModelCheckpoint])

def get_training_model_class(kind):
    if kind == 'default':
        return DefaultInpaintingTrainingModule

    raise ValueError(f'Unknown trainer module {kind}')


def make_training_model(config):
    kind = config.training_model.kind
    kwargs = dict(config.training_model)
    kwargs.pop('kind')
    kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'

    logging.info(f'Make training model {kind}')

    cls = get_training_model_class(kind)
    return cls(config, **kwargs)


def load_checkpoint(train_config, path, map_location='cuda', strict=True):
    model: torch.nn.Module = make_training_model(train_config)
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state['state_dict'], strict=strict)
    model.on_load_checkpoint(state)
    return model
