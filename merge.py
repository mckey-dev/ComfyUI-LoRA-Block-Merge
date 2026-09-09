"""LoRA state dict のパース、スケール、ランク連結 / SVD 圧縮によるマージ。"""

import torch

from .anima_block_weight import is_clip_key, key_multiplier

_PAIR_SPECS = (
    (".lora_B.default.weight", ".lora_A.default.weight", None),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight", None),
    (".lora_up.weight", ".lora_down.weight", ".lora_mid.weight"),
    (".lora_B.weight", ".lora_A.weight", None),
    ("_lora.up.weight", "_lora.down.weight", None),
    (".lora.up.weight", ".lora.down.weight", None),
    (".lora_B", ".lora_A", None),
)

_UNSUPPORTED = (
    ".dora_scale",
    "hada_w1_a",
    "hada_w1_b",
    "lokr_w1",
    "lokr_w2",
    "oft_blocks",
)


def _split_key(key):
    """state dict キーをモジュール名と up/down/alpha 等の種別に分解する。"""
    if key.endswith(".alpha"):
        return key[:-6], "alpha"
    if key.endswith(".dora_scale"):
        return key[:-11], "dora"
    for up_s, down_s, mid_s in _PAIR_SPECS:
        if key.endswith(up_s):
            return key[: -len(up_s)], "up"
        if key.endswith(down_s):
            return key[: -len(down_s)], "down"
        if mid_s is not None and key.endswith(mid_s):
            return key[: -len(mid_s)], "mid"
    return None, None


def _norm_module(name):
    """異なるキー接頭辞を同一モジュールとして束ねるための正規化名を返す。"""
    s = name
    if s.startswith("diffusion_model."):
        s = s[len("diffusion_model."):]
    elif s.startswith("lora_unet_"):
        s = s[len("lora_unet_"):]
    return s.replace(".", "_")


def parse_lora_modules(sd):
    """LoRA state dict をモジュール単位の up/down/alpha にまとめる。"""
    modules = {}
    for key, tensor in sd.items():
        for marker in _UNSUPPORTED:
            if marker in key:
                raise ValueError(f"Unsupported LoRA key: {key}")
        module, kind = _split_key(key)
        if module is None:
            continue
        entry = modules.setdefault(module, {
            "up": None,
            "down": None,
            "alpha": None,
            "mid": None,
            "dora": False,
            "up_suffix": None,
            "down_suffix": None,
        })
        if kind == "up":
            entry["up"] = tensor
            entry["up_suffix"] = key[len(module):]
        elif kind == "down":
            entry["down"] = tensor
            entry["down_suffix"] = key[len(module):]
        elif kind == "alpha":
            entry["alpha"] = float(tensor.item() if torch.is_tensor(tensor) else tensor)
        elif kind == "mid":
            entry["mid"] = tensor
        elif kind == "dora":
            entry["dora"] = True
    return modules


def _alpha_scale(alpha, rank):
    """alpha / rank のスケール係数を返す。alpha 未設定時は 1。"""
    if alpha is None:
        return 1.0
    return alpha / rank


def scaled_delta(up, down, alpha, strength, weight):
    """strength とブロック重みを掛けた LoRA delta（up @ down）を返す。"""
    rank = down.shape[0]
    scale = strength * weight * _alpha_scale(alpha, rank)
    return scale * (up.flatten(start_dim=1).float() @ down.flatten(start_dim=1).float())


def _extract_lora(diff, rank):
    """delta 行列を SVD して指定ランクの up/down に分解する。"""
    conv2d = diff.ndim == 4
    kernel_size = None if not conv2d else tuple(diff.shape[2:4])
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.shape[0:2]
    rank = min(int(rank), in_dim, out_dim)
    if rank < 1:
        raise ValueError("svd_rank must be >= 1")
    work = diff.float()
    if conv2d:
        work = work.flatten(start_dim=1) if conv2d_3x3 else work.squeeze(-1).squeeze(-1)
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    rank = min(rank, s.shape[0])
    u = u[:, :rank] @ torch.diag(s[:rank])
    vh = vh[:rank, :]
    if conv2d:
        u = u.reshape(out_dim, rank, 1, 1)
        vh = vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
    return u, vh


def _concat_scaled(parts):
    """複数 LoRA の up をスケールして連結し、down はそのまま連結する。"""
    ups = []
    downs = []
    for up, down, alpha, strength, weight in parts:
        rank = down.shape[0]
        scale = strength * weight * _alpha_scale(alpha, rank)
        ups.append(up.float() * scale)
        downs.append(down.float())
    cat_dim_up = 1
    cat_dim_down = 0
    return torch.cat(ups, dim=cat_dim_up), torch.cat(downs, dim=cat_dim_down)


def merge_state_dicts(loras, config, include_clip=False, compress_rank=False, svd_rank=16):
    """複数 LoRA をモジュール単位でマージした state dict を返す。"""
    grouped = {}
    for sd, strength in loras:
        modules = parse_lora_modules(sd)
        for module, entry in modules.items():
            if entry["dora"]:
                raise ValueError(f"DoRA is not supported: {module}")
            if entry["mid"] is not None:
                raise ValueError(f"LoCon mid weights are not supported: {module}")
            if entry["up"] is None or entry["down"] is None:
                continue
            sample_key = module + (entry["up_suffix"] or ".lora_up.weight")
            if not include_clip and is_clip_key(sample_key):
                continue
            weight = key_multiplier(sample_key, config)
            if weight == 0.0:
                continue
            norm = _norm_module(module)
            grouped.setdefault(norm, []).append({
                "module": module,
                "up": entry["up"],
                "down": entry["down"],
                "alpha": entry["alpha"],
                "strength": strength,
                "weight": weight,
                "up_suffix": entry["up_suffix"] or ".lora_up.weight",
                "down_suffix": entry["down_suffix"] or ".lora_down.weight",
            })

    if not grouped:
        raise ValueError("No matching LoRA modules to merge")

    output = {}
    for parts in grouped.values():
        first = parts[0]
        up_shape = first["up"].shape
        down_shape = first["down"].shape
        for part in parts[1:]:
            if part["up"].shape[0] != up_shape[0] or part["down"].shape[1:] != down_shape[1:]:
                raise ValueError(
                    f"Incompatible LoRA shapes for {first['module']}: "
                    f"{part['up'].shape}/{part['down'].shape} vs {up_shape}/{down_shape}"
                )
        up, down = _concat_scaled(
            (p["up"], p["down"], p["alpha"], p["strength"], p["weight"])
            for p in parts
        )
        if compress_rank:
            diff = up.flatten(start_dim=1) @ down.flatten(start_dim=1)
            if first["up"].ndim == 4:
                diff = diff.reshape(
                    first["up"].shape[0],
                    first["down"].shape[1],
                    first["down"].shape[2],
                    first["down"].shape[3],
                )
            if torch.max(torch.abs(diff)) == 0:
                continue
            concat_rank = down.shape[0]
            if svd_rank < concat_rank:
                up, down = _extract_lora(diff, svd_rank)
        new_rank = down.shape[0]
        dtype = first["up"].dtype
        module = first["module"]
        output[module + first["up_suffix"]] = up.to(dtype=dtype).contiguous().cpu()
        output[module + first["down_suffix"]] = down.to(dtype=dtype).contiguous().cpu()
        output[module + ".alpha"] = torch.tensor(float(new_rank))
    if not output:
        raise ValueError("Merged LoRA is empty")
    return output
