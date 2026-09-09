# LoRA Block Merge — ComfyUI Custom Nodes

更新日: 2026-09-10

Anima 向けの LoRA を 2〜5 個選び、1 つの `.safetensors` にまとめます。マージ時は Transformer のブロックごと、レイヤー種類（self-attn / cross-attn / MLP など）ごとに強さを変えられます。できたファイルは通常の LoRA と同じように読み込めます。

## ノード

**Anima LoRA Merge**

- プリセット: `all_enabled`, `character_detail`, `no_style_leak`, `face_focused`, `style_transfer`, `attn_only`, `mlp_only`, `custom`
- デフォルトプリセット: `style_transfer`
- LoRA スロット: 1〜2 は必須、3〜5 は任意（`None` = 未使用）
- `include_clip`: デフォルトオフ
- `compress_rank`: デフォルトオフ。オフ時は連結ランクを維持。オン時は SVD で `svd_rank` まで圧縮

保存先は ComfyUI の `loras` フォルダです。

## 典型的な使い方

例: LoRA 4 個、プリセット `style_transfer`、`strength_model=1`、`include_clip` オフ。

1. **Anima LoRA Merge** を 1 回キューする。上記 4 ファイル、`style_transfer`、strength `1`、`include_clip=false`、`compress_rank=false`
2. 保存されたファイルを通常の LoRA ローダーで strength `1` として読み込む

## インストール

このフォルダを `ComfyUI/custom_nodes/` にコピーします。

```
ComfyUI/custom_nodes/ComfyUI-LoRA-Block-Merge
```

ComfyUI を再起動すると、ノードは **LoRA Merge** に表示されます。

このフォルダで次を実行すると、マージ計算のテストが走ります。

```
python -m pytest
```
