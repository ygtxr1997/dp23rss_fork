from typing import Optional
import pathlib
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        pass

    def _get_accelerator(self):
        return getattr(self, "accelerator", None)

    def _is_main_model(self, key, value):
        return key == "model" and value is getattr(self, "model", None)

    def _is_stateful(self, value):
        if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
            return True
        if isinstance(value, (list, tuple)):
            return all(self._is_stateful(v) for v in value)
        return False

    def _state_dict_of(self, key, value, use_thread=True):
        accelerator = self._get_accelerator()

        # 主模型：统一保存底层原始权重
        if self._is_main_model(key, value) and accelerator is not None:
            state = accelerator.get_state_dict(value)
        elif hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
            state = value.state_dict()
        elif isinstance(value, (list, tuple)):
            state = [self._state_dict_of(f"{key}.{i}", v, use_thread=False) for i, v in enumerate(value)]
        else:
            raise TypeError(f"Object for key={key} is not stateful")

        if use_thread:
            state = _copy_to_cpu(state)
        return state

    def _load_state_dict_into(self, key, obj, state, **kwargs):
        accelerator = self._get_accelerator()

        if key == "model" and accelerator is not None:
            obj = accelerator.unwrap_model(obj)

        if hasattr(obj, "load_state_dict") and isinstance(state, dict):
            obj.load_state_dict(state, **kwargs)
            return

        if isinstance(obj, (list, tuple)) and isinstance(state, list):
            assert len(obj) == len(state), f"Length mismatch for {key}: {len(obj)} vs {len(state)}"
            for i, (sub_obj, sub_state) in enumerate(zip(obj, state)):
                self._load_state_dict_into(f"{key}.{i}", sub_obj, sub_state, **kwargs)
            return

        raise TypeError(f"Cannot load state for key={key}, obj={type(obj)}, state={type(state)}")

    def save_checkpoint(
            self,
            path=None,
            tag='latest',
            exclude_keys=None,
            include_keys=None,
            use_thread=False
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)

        accelerator = self._get_accelerator()

        # 如果上一次后台保存还没结束，先等它结束
        if self._saving_thread is not None:
            self._saving_thread.join()
            self._saving_thread = None

        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue

            if self._is_stateful(value):
                payload['state_dicts'][key] = self._state_dict_of(
                    key, value, use_thread=use_thread
                )
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)

        def _save():
            if accelerator is not None:
                # accelerate.save 会按其规则执行“每机一次/主进程一次”保存
                accelerator.save(payload, str(path))
            else:
                with path.open('wb') as f:
                    torch.save(payload, f, pickle_module=dill)

        # 不要在这里 wait_for_everyone()
        # 外层训练循环里在调用 save_checkpoint() 之前统一同步一次即可

        if accelerator is not None and not accelerator.is_main_process:
            return str(path.absolute())

        if use_thread:
            self._saving_thread = threading.Thread(target=_save)
            self._saving_thread.start()
        else:
            _save()

        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self._load_state_dict_into(key, self.__dict__[key], value, **kwargs)

        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(
        self,
        path=None,
        tag='latest',
        exclude_keys=None,
        include_keys=None,
        **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

        kwargs.setdefault("map_location", "cpu")

        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys
        )
        return payload

    @classmethod
    def create_from_checkpoint(
        cls,
        path,
        exclude_keys=None,
        include_keys=None,
        **kwargs
    ):
        kwargs.setdefault("map_location", "cpu")
        payload = torch.load(open(path, 'rb'), pickle_module=dill, **kwargs)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs
        )
        return instance

    def save_snapshot(self, tag='latest'):
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_copy_to_cpu(v) for v in x]
    else:
        return copy.deepcopy(x)