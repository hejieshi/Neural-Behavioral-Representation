import copy
import pickle
from types import SimpleNamespace

import torch

from neural_behavior.models import (PoseMixtureVAE, NeuralBehaviorMixtureVAE,
                               BehaviorLabelMixtureVAE, TransformerEncoder)


def build_model(config, normalization):
    shared = (config['frame_size'], config['latent_size'], 1, 1,
              normalization, config['num_experts'])
    if config['model'] == 'behavior':
        model = PoseMixtureVAE(*shared)
    elif config['model'] == 'label':
        model = BehaviorLabelMixtureVAE(*shared, num_labels=config['num_labels'])
    elif config['model'] == 'neural':
        model = NeuralBehaviorMixtureVAE(*shared)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    return model


def save_checkpoint(path, model):
    saved_model = copy.deepcopy(model).cpu()
    for name in ('data_max', 'data_min', 'data_avg', 'data_std'):
        value = getattr(saved_model, name, None)
        if isinstance(value, torch.Tensor):
            setattr(saved_model, name, value.detach().cpu())
    torch.save(saved_model, path)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renames = {
            'PoseNeuralEncoder': 'NeuralBehaviorEncoder',
            'PoseNeuralMixedDecoder': 'NeuralBehaviorMixedDecoder',
            'PoseNeuralMixtureVAE': 'NeuralBehaviorMixtureVAE',
            'PoseLabelEncoder': 'BehaviorLabelEncoder',
            'PoseLabelMixedDecoder': 'BehaviorLabelDecoder',
            'PoseLabelMixtureVAE': 'BehaviorLabelMixtureVAE',
        }
        if module in {'models', 'neural_behavior.models'}:
            module = 'neural_behavior.models'
            name = renames.get(name, name)
        return super().find_class(module, name)


def load_checkpoint(path, normalization):
    """Load a trusted full-model pickle, using freshly computed motion statistics."""
    pickle_module = SimpleNamespace(__name__='model_pickle', Unpickler=_LegacyUnpickler,
                                    load=pickle.load, loads=pickle.loads)
    model = torch.load(path, map_location='cpu', weights_only=False,
                       pickle_module=pickle_module)
    if not isinstance(model, (PoseMixtureVAE, NeuralBehaviorMixtureVAE, BehaviorLabelMixtureVAE)):
        raise ValueError('Expected a supported full-model checkpoint.')
    for module in model.modules():
        if isinstance(module, TransformerEncoder) and not hasattr(module, 'cls_pos_emb'):
            raise ValueError('Checkpoint is missing the saved CLS position vector.')
    model.mode = normalization['mode']
    for attr, key in [('data_max', 'max'), ('data_min', 'min'),
                      ('data_avg', 'avg'), ('data_std', 'std')]:
        setattr(model, attr, normalization[key])
    return model.eval()
