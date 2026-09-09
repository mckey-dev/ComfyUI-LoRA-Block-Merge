"""Anima LoRA を 2〜5 個マージして `.safetensors` に保存する ComfyUI ノード。"""

import os
import re

import comfy.utils
import folder_paths

from .anima_block_weight import (
    ANIMA_BLOCK_PRESETS,
    DEFAULT_BLOCK_VECTOR,
    LAYER_FLOAT_KEYS,
    resolve_block_config,
)
from .merge import merge_state_dicts

_UNUSED = ("", "None", "none")


def _load_lora_file(path):
    """LoRA ファイルを state dict として読み込む。"""
    return comfy.utils.load_torch_file(path, safe_load=True)


def _clean_name(name):
    """未使用スロットの名前を None に正規化する。"""
    if name is None:
        return None
    name = name.strip().strip('"').strip("'")
    if name in _UNUSED:
        return None
    return name


def _resolve_path(lora_name):
    """LoRA 名または絶対パスから実ファイルパスを返す。"""
    path = folder_paths.get_full_path("loras", lora_name)
    if path and os.path.isfile(path):
        return path
    if os.path.isabs(lora_name) and os.path.isfile(lora_name):
        return lora_name
    return None


def _safe_stem(name):
    """保存ファイル名に使える stem に整形する。"""
    name = os.path.basename(name.strip())
    if name.lower().endswith(".safetensors"):
        name = name[:-12]
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return name or "anima_lora_merge"


def _save_dir():
    """マージ結果の保存先（ComfyUI の loras フォルダ）を返す。"""
    paths = folder_paths.get_folder_paths("loras")
    if not paths:
        raise RuntimeError("No loras folder configured")
    return paths[0]


class AnimaLoraBlockMerge:
    """ブロック / レイヤー重み付きで Anima LoRA をマージするノード。"""

    @classmethod
    def INPUT_TYPES(cls):
        """ノード入力（LoRA スロット、プリセット、レイヤー重み）を定義する。"""
        lora_files = sorted(folder_paths.get_filename_list("loras"))
        required_loras = lora_files if lora_files else [""]
        optional_loras = ["None"] + lora_files
        layer_inputs = {
            key: ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05})
            for key in LAYER_FLOAT_KEYS
        }
        return {
            "required": {
                "lora_name_1": (required_loras,),
                "strength_1": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
                "lora_name_2": (required_loras,),
                "strength_2": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
                "lora_name_3": (optional_loras, {"default": "None"}),
                "strength_3": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
                "lora_name_4": (optional_loras, {"default": "None"}),
                "strength_4": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
                "lora_name_5": (optional_loras, {"default": "None"}),
                "strength_5": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                }),
                "include_clip": ("BOOLEAN", {"default": False}),
                "compress_rank": ("BOOLEAN", {"default": False}),
                "svd_rank": ("INT", {
                    "default": 16, "min": 1, "max": 512, "step": 1,
                }),
                "filename_prefix": ("STRING", {"default": "anima_style_merge"}),
                "preset": (list(ANIMA_BLOCK_PRESETS.keys()), {
                    "default": "style_transfer",
                }),
                **layer_inputs,
                "block_vector": ("STRING", {
                    "default": DEFAULT_BLOCK_VECTOR,
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "merge"
    CATEGORY = "LoRA Merge"
    OUTPUT_NODE = True

    def merge(
        self,
        lora_name_1,
        strength_1,
        lora_name_2,
        strength_2,
        lora_name_3,
        strength_3,
        lora_name_4,
        strength_4,
        lora_name_5,
        strength_5,
        include_clip,
        compress_rank,
        svd_rank,
        filename_prefix,
        preset,
        self_attn,
        cross_attn,
        mlp,
        adaln_self_attn,
        adaln_cross_attn,
        adaln_mlp,
        block_vector,
    ):
        """選択した LoRA をマージし、loras フォルダへ保存してパスを返す。"""
        selected = []
        for name, strength in (
            (lora_name_1, strength_1),
            (lora_name_2, strength_2),
            (lora_name_3, strength_3),
            (lora_name_4, strength_4),
            (lora_name_5, strength_5),
        ):
            name = _clean_name(name)
            if name is None:
                continue
            path = _resolve_path(name)
            if path is None:
                raise FileNotFoundError(f"LoRA not found: {name}")
            selected.append((name, path, strength))

        if len(selected) < 2 or len(selected) > 5:
            raise ValueError("Select 2 to 5 LoRAs")

        config = resolve_block_config(
            preset,
            {
                "self_attn": self_attn,
                "cross_attn": cross_attn,
                "mlp": mlp,
                "adaln_self_attn": adaln_self_attn,
                "adaln_cross_attn": adaln_cross_attn,
                "adaln_mlp": adaln_mlp,
            },
            block_vector,
        )
        loras = [(_load_lora_file(path), strength) for _, path, strength in selected]
        merged = merge_state_dicts(
            loras,
            config,
            include_clip=include_clip,
            compress_rank=compress_rank,
            svd_rank=svd_rank,
        )

        out_dir = _save_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, _safe_stem(filename_prefix) + ".safetensors")
        comfy.utils.save_torch_file(merged, out_path)
        return (out_path,)
