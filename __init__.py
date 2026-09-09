"""ComfyUI 向け Anima LoRA マージノードの登録。"""

try:
    from .anima_lora_merge import AnimaLoraBlockMerge
except ImportError as e:
    if "no known parent package" not in str(e):
        raise
    AnimaLoraBlockMerge = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
if AnimaLoraBlockMerge is not None:
    NODE_CLASS_MAPPINGS["AnimaLoraBlockMerge"] = AnimaLoraBlockMerge
    NODE_DISPLAY_NAME_MAPPINGS["AnimaLoraBlockMerge"] = "Anima LoRA Merge"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
