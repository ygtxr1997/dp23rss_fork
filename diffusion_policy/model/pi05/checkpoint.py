# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a JAX Pi0/Pi05 checkpoint to the OpenPI layout.

Reproduces the SigLIP / dual-expert-LLM / projection weight transforms exactly,
taking the model-shape fields as explicit arguments instead of looking them up in
an in-code config registry. The JAX parameter tree is loaded with a lazily
imported ``orbax`` / ``jax`` (kept out of module import so this package imports
without JAX installed and without depending on the external ``openpi`` package).

The converted weights are written as fp32 ``model.safetensors`` while the emitted
``config.json`` records a ``dtype: bfloat16`` hint for the eval loader.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

# So400m/14 SigLIP width and the two LLM expert widths (pi05 base).
_SIGLIP_WIDTH = 1152
_PALIGEMMA_WIDTH = 2048
_ACTION_WIDTH = 1024


def _load_jax_params(checkpoint_dir: str | pathlib.Path) -> dict:
    """Restore the JAX parameter pytree from ``{checkpoint_dir}/params`` as float32 numpy.

    ``orbax`` and ``jax`` are imported lazily here so the module imports without
    them; the JAX toolchain is only required to actually run a conversion.
    """
    import jax
    import orbax.checkpoint as ocp

    params_dir = pathlib.Path(checkpoint_dir) / "params"
    checkpointer = ocp.PyTreeCheckpointer()

    # The converter only needs host-side NumPy arrays.  Restoring without
    # restore args makes Orbax infer JAX arrays and their saved sharding.  That
    # fails for checkpoints produced on a different device topology (and in
    # particular can leave the deserializer with ``sharding=None``).  Build a
    # restore-args tree from the checkpoint metadata and explicitly request
    # NumPy leaves, which do not require device sharding.
    metadata = checkpointer.metadata(str(params_dir))
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
    )
    restored = checkpointer.restore(str(params_dir), restore_args=restore_args)
    # NNX checkpoints wrap parameter arrays as {"value": array}. Treat these
    # wrappers as leaves so all converters receive pure arrays, including SigLIP
    # and Gemma, rather than unwrapping only the action projections.
    restored = jax.tree_util.tree_map(
        lambda value_D: np.asarray(
            value_D["value"] if isinstance(value_D, dict) else value_D,
            dtype=np.float32,
        ),
        restored,
        is_leaf=lambda node: isinstance(node, dict) and set(node) == {"value"},
    )
    # Some orbax checkpoints (e.g. the pi05_base reference) wrap the parameter
    # tree under a top-level ``params`` collection; the converters expect the
    # unwrapped tree (``PaliGemma`` / ``action_in_proj`` / ... at the top), so
    # peel a single ``params`` wrapper when present.
    if "PaliGemma" not in restored and isinstance(restored.get("params"), dict):
        restored = restored["params"]
    return restored


def convert_siglip(params: dict) -> dict:
    """Convert SigLIP ViT parameters from JAX to the OpenPI layout."""
    pt: dict[str, torch.Tensor] = {}
    img = params["PaliGemma"]["img"]

    # Patch embedding (Conv2d): JAX (H, W, C_in, C_out) -> PT (C_out, C_in, H, W)
    pt["img.stem.weight"] = torch.from_numpy(
        img["embedding"]["kernel"].transpose(3, 2, 0, 1)
    )
    pt["img.stem.bias"] = torch.from_numpy(img["embedding"]["bias"])

    eb = img["Transformer"]["encoderblock"]
    ln0_scale_L_D, ln0_bias_L_D = eb["LayerNorm_0"]["scale"], eb["LayerNorm_0"]["bias"]
    ln1_scale_L_D, ln1_bias_L_D = eb["LayerNorm_1"]["scale"], eb["LayerNorm_1"]["bias"]
    dense0_kernel_L_D_M, dense0_bias_L_M = (
        eb["MlpBlock_0"]["Dense_0"]["kernel"],
        eb["MlpBlock_0"]["Dense_0"]["bias"],
    )
    dense1_kernel_L_M_D, dense1_bias_L_D = (
        eb["MlpBlock_0"]["Dense_1"]["kernel"],
        eb["MlpBlock_0"]["Dense_1"]["bias"],
    )

    mha = eb["MultiHeadDotProductAttention_0"]
    q_kernel_L_D_N_H, q_bias_L_N_H = mha["query"]["kernel"], mha["query"]["bias"]
    k_kernel_L_D_N_H, k_bias_L_N_H = mha["key"]["kernel"], mha["key"]["bias"]
    v_kernel_L_D_N_H, v_bias_L_N_H = mha["value"]["kernel"], mha["value"]["bias"]
    o_kernel_L_N_H_D, o_bias_L_D = mha["out"]["kernel"], mha["out"]["bias"]

    width = _SIGLIP_WIDTH
    for i in range(27):
        prefix = f"img.encoder.layers.{i}"
        pt[f"{prefix}.norm1.weight"] = torch.from_numpy(ln0_scale_L_D[i])
        pt[f"{prefix}.norm1.bias"] = torch.from_numpy(ln0_bias_L_D[i])
        pt[f"{prefix}.norm2.weight"] = torch.from_numpy(ln1_scale_L_D[i])
        pt[f"{prefix}.norm2.bias"] = torch.from_numpy(ln1_bias_L_D[i])

        # JAX Dense uses x @ kernel; PT Linear uses x @ weight.T, so weight = kernel.T.
        q_D_D = torch.from_numpy(q_kernel_L_D_N_H[i].reshape(width, width).T)
        k_D_D = torch.from_numpy(k_kernel_L_D_N_H[i].reshape(width, width).T)
        v_D_D = torch.from_numpy(v_kernel_L_D_N_H[i].reshape(width, width).T)
        q_D = torch.from_numpy(q_bias_L_N_H[i].reshape(width))
        k_D = torch.from_numpy(k_bias_L_N_H[i].reshape(width))
        v_D = torch.from_numpy(v_bias_L_N_H[i].reshape(width))
        pt[f"{prefix}.attn.in_proj_weight"] = torch.cat([q_D_D, k_D_D, v_D_D], dim=0)
        pt[f"{prefix}.attn.in_proj_bias"] = torch.cat([q_D, k_D, v_D], dim=0)

        pt[f"{prefix}.attn.out_proj.weight"] = torch.from_numpy(
            o_kernel_L_N_H_D[i].reshape(width, width).T
        )
        pt[f"{prefix}.attn.out_proj.bias"] = torch.from_numpy(o_bias_L_D[i])

        pt[f"{prefix}.mlp.fc1.weight"] = torch.from_numpy(dense0_kernel_L_D_M[i].T)
        pt[f"{prefix}.mlp.fc1.bias"] = torch.from_numpy(dense0_bias_L_M[i])
        pt[f"{prefix}.mlp.fc2.weight"] = torch.from_numpy(dense1_kernel_L_M_D[i].T)
        pt[f"{prefix}.mlp.fc2.bias"] = torch.from_numpy(dense1_bias_L_D[i])

    pt["img.encoder.norm.weight"] = torch.from_numpy(
        img["Transformer"]["encoder_norm"]["scale"]
    )
    pt["img.encoder.norm.bias"] = torch.from_numpy(
        img["Transformer"]["encoder_norm"]["bias"]
    )
    pt["img.head.weight"] = torch.from_numpy(img["head"]["kernel"].T)
    pt["img.head.bias"] = torch.from_numpy(img["head"]["bias"])
    pt["img.pos_embedding"] = torch.from_numpy(img["pos_embedding"])  # (1, 256, 1152)
    return pt


def convert_llm(params: dict, pi05: bool) -> dict:
    """Convert the dual-expert Gemma LLM (PaliGemma + action expert) from JAX to PyTorch."""
    pt: dict[str, torch.Tensor] = {}
    llm = params["PaliGemma"]["llm"]
    pt["llm.embedder.embedding.weight"] = torch.from_numpy(
        llm["embedder"]["input_embedding"]
    )

    layers = llm["layers"]
    pg_w, act_w = _PALIGEMMA_WIDTH, _ACTION_WIDTH

    q_einsum_L_N_D_H = layers["attn"]["q_einsum"]["w"]
    kv_einsum_L_2_K_D_H = layers["attn"]["kv_einsum"]["w"]
    o_einsum_L_N_H_D = layers["attn"]["attn_vec_einsum"]["w"]
    mlp_gating_L_2_D_M = layers["mlp"]["gating_einsum"]
    mlp_linear_L_M_D = layers["mlp"]["linear"]
    pre_attn_scale_L_D = layers["pre_attention_norm"]["scale"]
    pre_ffw_scale_L_D = layers["pre_ffw_norm"]["scale"]

    q_einsum_1_L_N_D_H = layers["attn"]["q_einsum_1"]["w"]
    kv_einsum_1_L_2_K_D_H = layers["attn"]["kv_einsum_1"]["w"]
    o_einsum_1_L_N_H_D = layers["attn"]["attn_vec_einsum_1"]["w"]
    mlp_gating_1_L_2_D_M = layers["mlp_1"]["gating_einsum"]
    mlp_linear_1_L_M_D = layers["mlp_1"]["linear"]

    n_layers = q_einsum_L_N_D_H.shape[0]
    for i in range(n_layers):
        # Expert 0 (PaliGemma)
        pt[f"llm.layers.{i}.attn.q_proj.0.weight"] = torch.from_numpy(
            q_einsum_L_N_D_H[i].transpose(0, 2, 1).reshape(pg_w, pg_w)
        )
        pt[f"llm.layers.{i}.attn.k_proj.0.weight"] = torch.from_numpy(
            kv_einsum_L_2_K_D_H[i, 0, 0].T
        )
        pt[f"llm.layers.{i}.attn.v_proj.0.weight"] = torch.from_numpy(
            kv_einsum_L_2_K_D_H[i, 1, 0].T
        )
        pt[f"llm.layers.{i}.attn.o_proj.0.weight"] = torch.from_numpy(
            o_einsum_L_N_H_D[i].reshape(pg_w, pg_w).T
        )
        pt[f"llm.layers.{i}.pre_attention_norms.0.scale"] = torch.from_numpy(
            pre_attn_scale_L_D[i]
        )
        pt[f"llm.layers.{i}.pre_ffw_norms.0.scale"] = torch.from_numpy(pre_ffw_scale_L_D[i])
        pt[f"llm.layers.{i}.mlps.0.w_gating"] = torch.from_numpy(mlp_gating_L_2_D_M[i])
        pt[f"llm.layers.{i}.mlps.0.w_linear"] = torch.from_numpy(mlp_linear_L_M_D[i])

        # Expert 1 (Action Expert)
        pt[f"llm.layers.{i}.attn.q_proj.1.weight"] = torch.from_numpy(
            q_einsum_1_L_N_D_H[i].transpose(0, 2, 1).reshape(pg_w, act_w)
        )
        pt[f"llm.layers.{i}.attn.k_proj.1.weight"] = torch.from_numpy(
            kv_einsum_1_L_2_K_D_H[i, 0, 0].T
        )
        pt[f"llm.layers.{i}.attn.v_proj.1.weight"] = torch.from_numpy(
            kv_einsum_1_L_2_K_D_H[i, 1, 0].T
        )
        pt[f"llm.layers.{i}.attn.o_proj.1.weight"] = torch.from_numpy(
            o_einsum_1_L_N_H_D[i].reshape(pg_w, act_w).T
        )
        pt[f"llm.layers.{i}.mlps.1.w_gating"] = torch.from_numpy(mlp_gating_1_L_2_D_M[i])
        pt[f"llm.layers.{i}.mlps.1.w_linear"] = torch.from_numpy(mlp_linear_1_L_M_D[i])

        if pi05:
            pre_attn_1 = layers["pre_attention_norm_1"]
            pre_ffw_1 = layers["pre_ffw_norm_1"]
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.weight"] = (
                torch.from_numpy(pre_attn_1["Dense_0"]["kernel"][i].T)
            )
            pt[f"llm.layers.{i}.pre_attention_norms.1.ada_modulation.bias"] = (
                torch.from_numpy(pre_attn_1["Dense_0"]["bias"][i])
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.weight"] = (
                torch.from_numpy(pre_ffw_1["Dense_0"]["kernel"][i].T)
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.ada_modulation.bias"] = (
                torch.from_numpy(pre_ffw_1["Dense_0"]["bias"][i])
            )
        else:
            pt[f"llm.layers.{i}.pre_attention_norms.1.scale"] = torch.from_numpy(
                layers["pre_attention_norm_1"]["scale"][i]
            )
            pt[f"llm.layers.{i}.pre_ffw_norms.1.scale"] = torch.from_numpy(
                layers["pre_ffw_norm_1"]["scale"][i]
            )

    pt["llm.final_norms.0.scale"] = torch.from_numpy(llm["final_norm"]["scale"])
    if pi05:
        final_norm_1 = llm["final_norm_1"]
        pt["llm.final_norms.1.ada_modulation.weight"] = torch.from_numpy(
            final_norm_1["Dense_0"]["kernel"].T
        )
        pt["llm.final_norms.1.ada_modulation.bias"] = torch.from_numpy(
            final_norm_1["Dense_0"]["bias"]
        )
    else:
        pt["llm.final_norms.1.scale"] = torch.from_numpy(llm["final_norm_1"]["scale"])
    return pt


def convert_projections(params: dict, pi05: bool) -> dict:
    """Convert action/time projection parameters from JAX to PyTorch."""
    pt: dict[str, torch.Tensor] = {}
    if pi05:
        proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        proj_keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    for key in proj_keys:
        if key not in params:
            continue
        kernel_D_M = params[key]["kernel"]
        bias_M = params[key]["bias"]
        pt[f"{key}.weight"] = torch.from_numpy(np.array(kernel_D_M).T)
        pt[f"{key}.bias"] = torch.from_numpy(np.array(bias_M))
    return pt


def sha256_file(path):
    """Hash an asset in bounded-memory blocks."""
    import hashlib
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_weights(model, path):
    """Load base weights, zero-initializing only additive foveation parameters."""
    from safetensors.torch import load_file
    path = pathlib.Path(path).expanduser()
    if path.is_dir():
        path = path / 'model.safetensors'
    state = load_file(str(path), device='cpu')
    # This migration is only for base initialization, never training/inference resume.
    # Explicit zeros also work when loading into a previously used model instance.
    if model.img.level_embedding is not None and 'img.level_embedding' not in state:
        state['img.level_embedding'] = torch.zeros_like(model.img.level_embedding, device='cpu')
    if model.img.apt_position is not None and 'img.apt_position' not in state:
        state['img.apt_position'] = torch.zeros_like(model.img.apt_position, device='cpu')
    if model.img.patch_level_adapters is not None:
        for index, adapter in enumerate(model.img.patch_level_adapters):
            for name, parameter in adapter.state_dict().items():
                key = f'img.patch_level_adapters.{index}.{name}'
                if key not in state:
                    state[key] = torch.zeros_like(parameter, device='cpu')
    model.load_state_dict(state, strict=True)
    return path


def convert_checkpoint(input_model, output_model, tokenizer_path, *, norm_stats_path=None, source_uri):
    """Convert official JAX pi0.5 weights and record source/output SHA256 manifests."""
    import json
    import shutil
    import sys
    from dataclasses import asdict
    from importlib.metadata import version
    from safetensors.torch import save_file
    from .config import Pi05Config
    from .model import Pi05Model

    input_model, output_model = pathlib.Path(input_model).resolve(), pathlib.Path(output_model).resolve()
    if output_model == input_model or input_model in output_model.parents:
        raise ValueError('Write converted assets outside the source checkpoint directory')
    output_model.mkdir(parents=True, exist_ok=False)
    params = _load_jax_params(input_model)
    state = {}
    for part in (convert_siglip(params), convert_llm(params, True), convert_projections(params, True)):
        state.update({key: value_D.contiguous() for key, value_D in part.items()})
    # Verify all shapes without allocating another full model on CPU.
    with torch.device('meta'):
        expected = Pi05Model(Pi05Config()).state_dict()
    if set(state) != set(expected):
        raise ValueError(f'Converted keys differ: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for key, value_D in state.items():
        if value_D.shape != expected[key].shape:
            raise ValueError(f'{key}: converted {value_D.shape}, expected {expected[key].shape}')
    save_file(state, str(output_model / 'model.safetensors'))
    shutil.copyfile(tokenizer_path, output_model / 'tokenizer.model')
    if norm_stats_path is not None:
        shutil.copyfile(norm_stats_path, output_model / 'norm_stats.json')
    (output_model / 'config.json').write_text(json.dumps(asdict(Pi05Config()), indent=2) + '\n')
    sources = json.loads(pathlib.Path(__file__).with_name('sources.json').read_text())
    manifest = {
        'source_uri': source_uri, 'sources': sources, 'conversion_argv': sys.argv,
        'conversion': 'official JAX pi05 -> RLinf bare FP32 safetensors; LIBERO horizon=10',
        'versions': {name: version(name) for name in ('torch', 'numpy', 'jax', 'orbax-checkpoint', 'safetensors')},
        'input_sha256': {str(path.relative_to(input_model)): sha256_file(path)
                         for path in sorted(input_model.rglob('*')) if path.is_file()},
        'output_sha256': {path.name: sha256_file(path) for path in sorted(output_model.iterdir()) if path.is_file()},
        'normalization': 'explicit supplied asset' if norm_stats_path else 'not supplied; fit project training split before SFT',
        'numerical_alignment': 'NOT RUN',
    }
    (output_model / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output_model
