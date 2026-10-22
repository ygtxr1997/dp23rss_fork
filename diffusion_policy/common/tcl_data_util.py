"""Shared validation for TCL dataset and checkpoint normalization assets."""

import copy
import json
from pathlib import Path

import numpy as np


def _pair(low, high, dim, label):
    low_D, high_D = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    if low_D.shape != (dim,) or high_D.shape != (dim,):
        raise ValueError(f'{label}: expected two vectors of length {dim}')
    if not np.isfinite(low_D).all() or not np.isfinite(high_D).all() or np.any(high_D < low_D):
        raise ValueError(f'{label}: bounds must be finite and ordered')
    return low_D, high_D


def require_same_bounds(left, right, label):
    """Reject conflicting statistics after conversion to training float32 precision."""
    if any(not np.array_equal(a_D, b_D) for a_D, b_D in zip(left, right)):
        raise ValueError(f'{label}: normalization statistics differ')


def action_bounds(metadata, action_dim=7, required=True):
    """Read canonical or legacy DP bounds, checking every supplied representation."""
    statistics = metadata.get('statistics', {})
    candidates = [(statistics.get('actions', {}), 'min', 'max'),
                  (statistics, 'action_min', 'action_max'),
                  (statistics.get('rel_actions', {}), 'min', 'max'),
                  (metadata, 'action_min', 'action_max')]
    result = None
    for container, low_key, high_key in candidates:
        if low_key not in container and high_key not in container:
            continue
        pair = _pair(container.get(low_key), container.get(high_key), action_dim, 'TCL action min/max')
        if result is not None:
            require_same_bounds(result, pair, 'TCL new/legacy action bounds')
        result = pair
    if result is None and required:
        raise ValueError('TCL metadata is missing action min/max')
    return result


def quantile_stats(statistics, state_dim, action_dim=7, required=True):
    """Validate complete TCL quantiles; only entirely absent quantiles may fall back."""
    present = any(q in statistics.get(key, {}) for key in ('state', 'actions') for q in ('q01', 'q99'))
    if not present and not required:
        return None
    for key, dim in (('state', state_dim), ('actions', action_dim)):
        item = statistics.get(key, {})
        _pair(item.get('q01'), item.get('q99'), dim, f'TCL {key} q01/q99')
    return copy.deepcopy(statistics)


def load_tcl_quantiles(metadata, asset_dir, state_dim):
    """Prefer metadata quantiles, falling back only for legacy assets without them."""
    statistics = quantile_stats(metadata.get('statistics', {}), state_dim, required=False)
    source = 'dataset_metadata.json'
    if statistics is None:
        path = Path(asset_dir).expanduser() / 'norm_stats.json'
        payload = json.loads(path.read_text())
        statistics = quantile_stats(payload.get('norm_stats', payload), state_dim)
        source = str(path)
    return statistics, source


DP_NORM = dict(state='identity', action='minmax', force='quantile')
PI05_NORM = dict(state='quantile', action='quantile', force='quantile')


def configure_dp_action_clipping(cfg):
    """Disable DP sample clipping when action normalization uses quantiles."""
    dataset = cfg.get('task', {}).get('dataset', {})
    norm = dataset.get('norm')
    action_method = norm.get('action') if norm is not None else None
    if action_method == 'quantile':
        scheduler = cfg.get('policy', {}).get('noise_scheduler')
        if scheduler is not None and scheduler.get('clip_sample') is not False:
            # Quantile-normalized actions may legitimately lie outside [-1, 1].
            scheduler['clip_sample'] = False
            print('[DP config] action quantile normalization: clip_sample=false', flush=True)
    return cfg


def norm_methods(norm, defaults):
    """Resolve supported method names, accepting the historical mean alias."""
    methods = dict(defaults)
    if norm is not None:
        if set(norm) - set(methods):
            raise ValueError(f'Unknown normalization keys: {set(norm) - set(methods)}')
        methods.update(norm)
    methods = {key: 'mean_std' if value == 'mean' else value for key, value in methods.items()}
    if any(value not in ('identity', 'minmax', 'quantile', 'mean_std') for value in methods.values()):
        raise ValueError(f'Unsupported normalization methods: {methods}')
    return methods


def canonical_statistics(metadata):
    """Adapt old TCL names without changing the stored source/split metadata."""
    statistics = copy.deepcopy(metadata.get('statistics', {}))
    bounds = action_bounds(metadata, required=False)
    if bounds is not None:
        statistics.setdefault('actions', {}).update(min=bounds[0].tolist(), max=bounds[1].tolist())
    for key, old_key in (('state', 'robot_obs'), ('actions', 'rel_actions')):
        for name, value in statistics.get(old_key, {}).items():
            statistics.setdefault(key, {}).setdefault(name, value)
    for item in statistics.values():
        if isinstance(item, dict):
            for name, old_name in (('q01', 'p01'), ('q99', 'p99')):
                if old_name in item:
                    item.setdefault(name, item[old_name])
    return statistics


def make_normalization(metadata, norm=None, *, defaults=None, dimensions=None,
                       saved=None, legacy_single=False):
    """Build or validate a self-contained normalization contract for active fields."""
    defaults = DP_NORM if defaults is None else defaults
    methods = norm_methods(saved['methods'] if norm is None and saved is not None else norm, defaults)
    dimensions = dict(state=6, action=7) if dimensions is None else dimensions
    statistics = canonical_statistics(metadata)
    fields = {}
    for key, dim in dimensions.items():
        method = methods[key]
        item = statistics.get({'action': 'actions', 'force': 'force_torque'}.get(key, key), {})
        names = {'identity': (), 'minmax': ('min', 'max'),
                 'quantile': ('q01', 'q99'), 'mean_std': ('mean', 'std')}[method]
        selected = {}
        for name in names:
            values_D = np.asarray(item.get(name), dtype=np.float32)
            if values_D.ndim != 1 or len(values_D) < dim or not np.isfinite(values_D).all():
                raise ValueError(f'TCL {key}.{name}: need {dim} finite statistics; prepare a new asset directory')
            selected[name] = values_D[:dim].tolist()
        if method in ('minmax', 'quantile'):
            _pair(*(selected[name] for name in names), dim, f'TCL {key} {method}')
        if method == 'mean_std' and np.any(np.asarray(selected['std']) < 0):
            raise ValueError(f'TCL {key}.std must be nonnegative')
        fields[key] = dict(method=method, statistics=selected, dim=dim,
                           epsilon=1e-6, denominator='add' if method == 'quantile' else 'clamp',
                           clip=False)
        if legacy_single:
            fields[key].update(epsilon=0.0, denominator='clamp', clip=method == 'quantile')
    contract = dict(version=1, methods=methods, fields=fields)
    if saved is not None:
        if saved.get('version') != 1:
            raise ValueError('Unsupported TCL normalization contract version')
        if methods != saved['methods']:
            raise ValueError(f'TCL norm config differs from checkpoint: {methods} != {saved["methods"]}')
        # A saved contract owns numerical semantics (including old clipping/epsilon).
        for key, field in fields.items():
            previous = saved['fields'][key]
            if (not np.isfinite(previous['epsilon']) or previous['epsilon'] < 0
                    or previous['denominator'] not in ('add', 'clamp')
                    or type(previous['clip']) is not bool):
                raise ValueError(f'Invalid saved TCL {key} numerical normalization contract')
            if field['dim'] != previous['dim'] or field['method'] != previous['method']:
                raise ValueError(f'TCL {key} normalization contract differs from checkpoint')
            for name, value in field['statistics'].items():
                require_same_bounds((np.asarray(value, dtype=np.float32),),
                                    (np.asarray(previous['statistics'][name], dtype=np.float32),),
                                    f'TCL checkpoint {key}.{name}')
        return copy.deepcopy(saved)
    return contract


def transform_values(values_N_D, field, *, inverse=False):
    """Apply one saved field contract to NumPy or Torch values with trailing D."""
    # Import lazily so metadata inspection does not require the training stack.
    import torch
    is_tensor = torch.is_tensor(values_N_D)
    values_N_D = values_N_D.float() if is_tensor else np.asarray(values_N_D, dtype=np.float32)
    dim = values_N_D.shape[-1]
    if dim > field['dim']:
        raise ValueError(f'Input dimension {dim} exceeds normalization contract {field["dim"]}')
    method = field['method']
    if method == 'identity':
        return values_N_D
    def vector(name):
        values_D = field['statistics'][name][:dim]
        return (torch.as_tensor(values_D, dtype=torch.float32, device=values_N_D.device)
                if is_tensor else np.asarray(values_D, dtype=np.float32))
    if method == 'mean_std':
        offset_D, scale_D = vector('mean'), vector('std')
    else:
        low_name, high_name = ('min', 'max') if method == 'minmax' else ('q01', 'q99')
        offset_D, high_D = vector(low_name), vector(high_name)
        scale_D = high_D - offset_D
        if field['clip'] and not inverse:
            values_N_D = (values_N_D.clamp(offset_D, high_D) if is_tensor
                          else np.clip(values_N_D, offset_D, high_D))
    epsilon = field['epsilon']
    scale_D = (scale_D + epsilon if field['denominator'] == 'add' else
               scale_D.clamp_min(epsilon) if is_tensor else np.maximum(scale_D, epsilon))
    if inverse:
        unit_N_D = values_N_D if method == 'mean_std' else (values_N_D + 1) / 2
        return unit_N_D * scale_D + offset_D
    unit_N_D = (values_N_D - offset_D) / scale_D
    return unit_N_D if method == 'mean_std' else unit_N_D * 2 - 1


def load_tcl_statistics(metadata, asset_dir, norm, state_dim):
    """Load selected statistics, allowing only the established legacy quantile fallback."""
    methods = norm_methods(norm, PI05_NORM)
    statistics = canonical_statistics(metadata)
    source = 'dataset_metadata.json'
    if 'quantile' in (methods['state'], methods['action']):
        # An incomplete quantile block is always an error, even in legacy assets.
        quantiles = quantile_stats(statistics, state_dim, required=False)
        if quantiles is None:
            quantiles, source = load_tcl_quantiles(metadata, asset_dir, state_dim)
            for key, item in quantiles.items():
                statistics.setdefault(key, {}).update(item)
    return statistics, source


def checkpoint_normalization(metadata, assets, norm, state_dim):
    """Validate Pi0.5 TCL metadata/config against the weights' saved normalization."""
    saved_stats = assets['norm_stats'].get('norm_stats', assets['norm_stats'])
    saved = assets.get('normalization')
    if saved is None:
        saved = make_normalization({'statistics': saved_stats}, defaults=PI05_NORM,
                                   dimensions=dict(state=state_dim, action=7))
    # Build from checkpoint stats first; do not let metadata replace trained parameters.
    saved = make_normalization({'statistics': saved_stats}, norm, defaults=PI05_NORM,
                               dimensions=dict(state=state_dim, action=7), saved=saved)
    current = canonical_statistics(metadata)
    for key, field in saved['fields'].items():
        item = current.setdefault('actions' if key == 'action' else key, {})
        for name, value in field['statistics'].items():
            if name in item:
                require_same_bounds((np.asarray(item[name], dtype=np.float32)[:field['dim']],),
                                    (np.asarray(value, dtype=np.float32),), f'TCL metadata {key}.{name}')
            item[name] = copy.deepcopy(value)
    if metadata.get('normalization') is not None and metadata['normalization'] != saved:
        raise ValueError('TCL dataset and policy normalization contracts differ')
    metadata['statistics'] = current
    metadata['normalization'] = copy.deepcopy(saved)
    return saved
