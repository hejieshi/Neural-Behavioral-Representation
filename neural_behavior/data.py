import numpy as np
import torch
from scipy.io import loadmat

def load_motion(path):
    with np.load(path) as archive:
        data = np.asarray(archive['data'], dtype=np.float32)
        ends = np.asarray(archive['end_indices'], dtype=np.int64).reshape(-1)
    if data.ndim != 2 or data.shape[1] != 151 or not np.isfinite(data).all():
        raise ValueError('mocap data must be finite with shape (frames, 151).')
    if len(ends) == 0 or np.any(np.diff(ends) <= 0) or ends[-1] != len(data) - 1:
        raise ValueError('end_indices must contain increasing, inclusive clip ends.')
    return torch.from_numpy(data), ends


def load_neural(path):
    features = loadmat(path)['features']
    if features.ndim != 3 or features.shape[:2] != (62, 30):
        raise ValueError('ECoG features must have shape (62, 30, timepoints).')
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('ECoG features contain non-finite values.')
    return torch.from_numpy(features)


def normalize_neural(data, mean, std, mode='zscore'):
    """Apply the selected transform with raw-feature statistics."""
    if mode not in {'zscore', 'legacy-double'}:
        raise ValueError(f'Unknown neural normalization: {mode}')
    z = (data - mean) / std
    return (z - std) / std if mode == 'legacy-double' else z


def load_labels(path, num_labels=16):
    labels = np.load(path, allow_pickle=False)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Labels must be a one-dimensional integer array aligned with mocap rows.')
    if len(labels) == 0 or labels.min() < 0 or labels.max() >= num_labels:
        raise ValueError(f'Label IDs must be in [0, {num_labels - 1}].')
    return torch.nn.functional.one_hot(torch.from_numpy(labels.astype(np.int64)), num_labels).float()


def valid_starts(length, end_indices, rollout):
    """One condition frame followed by rollout targets, without crossing a clip."""
    starts = np.arange(max(0, length - rollout))
    valid = np.ones(len(starts), dtype=bool)
    for end in end_indices:
        valid &= ~((starts <= end) & (starts + rollout > end))
    return starts[valid]


def prepare_neural(data, mode='legacy-double'):
    """Fit each channel/band over the complete file, before any time slicing."""
    if len(data) < 2:
        raise ValueError('At least two neural timepoints are required.')
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True, unbiased=True) + 1e-8
    return normalize_neural(data, mean, std, mode)


def motion_normalization(data):
    """Compute motion normalization from the complete mocap file."""
    std = data.std(dim=0)
    std[std == 0] = 1
    return dict(mode='zscore', max=data.max(dim=0).values,
                min=data.min(dim=0).values, avg=data.mean(dim=0), std=std)
