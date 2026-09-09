"""Anima のブロック / レイヤー重みプリセットとキー倍率の解決。"""

import re

NUM_ANIMA_BLOCKS = 28

ANIMA_BLOCK_PRESETS = {
    "all_enabled": {
        "self_attn": 1.0,
        "cross_attn": 1.0,
        "mlp": 1.0,
        "adaln_self_attn": 1.0,
        "adaln_cross_attn": 1.0,
        "adaln_mlp": 1.0,
    },
    "character_detail": {
        "self_attn": 1.0,
        "cross_attn": 0.0,
        "mlp": 1.0,
        "adaln_self_attn": 0.8,
        "adaln_cross_attn": 0.0,
        "adaln_mlp": 0.8,
    },
    "no_style_leak": {
        "self_attn": 1.0,
        "cross_attn": 0.0,
        "mlp": 0.8,
        "adaln_self_attn": 0.5,
        "adaln_cross_attn": 0.0,
        "adaln_mlp": 0.5,
    },
    "face_focused": {
        "self_attn": 1.0,
        "cross_attn": 0.3,
        "mlp": 1.0,
        "adaln_self_attn": 0.6,
        "adaln_cross_attn": 0.1,
        "adaln_mlp": 0.6,
    },
    "style_transfer": {
        "self_attn": 0.2,
        "cross_attn": 1.0,
        "mlp": 0.3,
        "adaln_self_attn": 0.2,
        "adaln_cross_attn": 1.0,
        "adaln_mlp": 0.3,
    },
    "attn_only": {
        "self_attn": 1.0,
        "cross_attn": 1.0,
        "mlp": 0.0,
        "adaln_self_attn": 0.0,
        "adaln_cross_attn": 0.0,
        "adaln_mlp": 0.0,
    },
    "mlp_only": {
        "self_attn": 0.0,
        "cross_attn": 0.0,
        "mlp": 1.0,
        "adaln_self_attn": 0.0,
        "adaln_cross_attn": 0.0,
        "adaln_mlp": 1.0,
    },
    "custom": None,
}

LAYER_FLOAT_KEYS = (
    "self_attn",
    "cross_attn",
    "mlp",
    "adaln_self_attn",
    "adaln_cross_attn",
    "adaln_mlp",
)

DEFAULT_BLOCK_VECTOR = ",".join(["1.0"] * NUM_ANIMA_BLOCKS)

ANIMA_BLOCK_RE = re.compile(
    r"blocks[_.](\d+)[_.]"
    r"(adaln_modulation_self_attn|adaln_modulation_cross_attn|adaln_modulation_mlp"
    r"|self_attn|cross_attn|mlp)"
)

LAYER_TYPE_MAP = {
    "self_attn": "self_attn",
    "cross_attn": "cross_attn",
    "mlp": "mlp",
    "adaln_modulation_self_attn": "adaln_self_attn",
    "adaln_modulation_cross_attn": "adaln_cross_attn",
    "adaln_modulation_mlp": "adaln_mlp",
}

CLIP_RE = re.compile(
    r"(^|[_./])(lora_te\d*|text_encoders?|qwen3|t5xxl|clip_[lgh]|cond_stage)([_./]|$)",
    re.IGNORECASE,
)


def parse_block_vector(text):
    """カンマ区切りのブロック重みを 28 要素のリストに正規化する。"""
    try:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        bw = [float(p) for p in parts]
    except ValueError:
        bw = [1.0] * NUM_ANIMA_BLOCKS
    if len(bw) < NUM_ANIMA_BLOCKS:
        bw.extend([1.0] * (NUM_ANIMA_BLOCKS - len(bw)))
    return bw[:NUM_ANIMA_BLOCKS]


def resolve_block_config(preset, layer_floats, block_vector):
    """プリセットまたは custom のレイヤー重みとブロック重みを解決する。"""
    if preset != "custom" and preset in ANIMA_BLOCK_PRESETS and ANIMA_BLOCK_PRESETS[preset] is not None:
        layer_weights = ANIMA_BLOCK_PRESETS[preset].copy()
    else:
        layer_weights = {k: float(layer_floats[k]) for k in LAYER_FLOAT_KEYS}
    return {
        "layer_weights": layer_weights,
        "block_weights": parse_block_vector(block_vector),
    }


def is_clip_key(key):
    """テキストエンコーダ（CLIP）系の LoRA キーなら True。"""
    return CLIP_RE.search(key) is not None


def key_multiplier(key, config):
    """キーに対応するブロック重み × レイヤー重みを返す。"""
    m = ANIMA_BLOCK_RE.search(key)
    if not m:
        return 1.0
    idx = int(m.group(1))
    raw_type = m.group(2)
    norm_type = LAYER_TYPE_MAP.get(raw_type, raw_type)
    block_v = config["block_weights"]
    bw = block_v[idx] if idx < len(block_v) else 1.0
    lw = config["layer_weights"].get(norm_type, 1.0)
    return round(bw * lw, 4)
