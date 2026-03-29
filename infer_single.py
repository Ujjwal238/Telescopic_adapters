"""
Single-image inference script for CLIPSegDenseAdapter.
No dataset, annotation JSON, or project imports required.

Usage:
    python infer_single.py
    (Edit the CONFIG block below before running)
"""

# ============================================================
#  CONFIG — Edit these before running
# ============================================================

IMAGE_PATH      = "/content/poly.png"
PROMPT          = "pink polyp"
CKPT_PATH       = "/content/drive/MyDrive/Medical_checkpoint/best.ckpt"
OUTPUT_PATH     = "output_mask.png"

# Model settings — must match what the checkpoint was trained with
CLIPSEG_HF_API  = "CIDAS/clipseg-rd64-refined"
ADAPTER_DIM     = 64        # Check your training YAML to confirm this value
ADAPTER_IN_V    = True
ADAPTER_IN_L    = True
ADAPTER_IN_COND = True

# Inference settings
IMG_SIZE        = [352, 352]
IMG_MEAN        = [0.485, 0.456, 0.406]
IMG_STD         = [0.229, 0.224, 0.225]
CONTEXT_LENGTH  = 77
DEVICE          = "cuda"    # "cpu" if no GPU available
PRECISION       = "fp16"    # "fp16" or "fp32"

# ============================================================

import os
from typing import Optional, List

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import CLIPSegForImageSegmentation
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)


# ============================================================
#  Model definition (from clipseg_adapter.py)
# ============================================================

class Adapter(nn.Module):
    def __init__(self, input_dim: int, adapter_dim: int, use_gelu=False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        bottleneck_dim = max(8, min(adapter_dim, input_dim // 4))
        self.down_project = nn.Linear(self.input_dim, bottleneck_dim, bias=False)
        self.up_project = nn.Linear(bottleneck_dim, self.input_dim, bias=False)
        self.activation = nn.GELU() if use_gelu else nn.SiLU()
        self.norm = nn.LayerNorm(bottleneck_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.down_project(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_project(h)
        return residual + self.scale * h


class CLIPSegDenseAdapter(nn.Module):
    def __init__(
        self,
        clipseg_hf_api: str,
        adapter_dim: int,
        freeze_clipseg: bool = True,
        adapter_in_v: bool = True,
        adapter_in_l: bool = True,
        adapter_in_cond: bool = True,
    ) -> None:
        super().__init__()

        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api)
        self.adapter_dim = adapter_dim
        self.adapter_in_v = adapter_in_v
        self.adapter_in_l = adapter_in_l
        self.adapter_in_cond = adapter_in_cond
        self.clipseg_config = self.clipseg.config
        self.clipseg.requires_grad_(not freeze_clipseg)

        if self.adapter_in_v:
            max_extract_layer = max(self.clipseg_config.extract_layers)
            self.v_attn_adapters = nn.ModuleList()
            self.v_out_adapters = nn.ModuleList()
            for i in range(max_extract_layer):
                layer_factor = (i + 1) / max_extract_layer
                current_adapter_dim = max(8, int(adapter_dim * layer_factor * 0.5))
                self.v_attn_adapters.append(Adapter(
                    input_dim=self.clipseg_config.vision_config.hidden_size,
                    adapter_dim=current_adapter_dim,
                ))
                self.v_out_adapters.append(Adapter(
                    input_dim=self.clipseg_config.vision_config.hidden_size,
                    adapter_dim=current_adapter_dim,
                ))

        if self.adapter_in_l:
            max_extract_layer = max(self.clipseg_config.extract_layers)
            num_text_adapters = min(3, max_extract_layer)
            self.l_attn_adapters = nn.ModuleList()
            self.l_out_adapters = nn.ModuleList()
            for i in range(num_text_adapters):
                text_adapter_dim = max(8, adapter_dim // 4)
                self.l_attn_adapters.append(Adapter(
                    input_dim=self.clipseg_config.text_config.hidden_size,
                    adapter_dim=text_adapter_dim,
                ))
                self.l_out_adapters.append(Adapter(
                    input_dim=self.clipseg_config.text_config.hidden_size,
                    adapter_dim=text_adapter_dim,
                ))

        if self.adapter_in_cond:
            self.cond_adapter = Adapter(
                input_dim=self.clipseg_config.projection_dim,
                adapter_dim=max(16, adapter_dim // 8),
            )

        self.boundary_enhancer = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.SiLU(),
            nn.Conv2d(4, 1, 1, bias=False),
        )

    def vision_forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        clip_vision_model = self.clipseg.clip.vision_model
        encoder_state = clip_vision_model.embeddings(pixel_values)
        encoder_state = clip_vision_model.pre_layrnorm(encoder_state)

        encoder_hidden_states = ()
        max_adapter_layers = len(self.v_attn_adapters) if self.adapter_in_v else 0

        for idx, encoder_layer in enumerate(clip_vision_model.encoder.layers):
            encoder_hidden_states = encoder_hidden_states + (encoder_state,)
            residual = encoder_state
            encoder_state = encoder_layer.layer_norm1(encoder_state)
            encoder_state, _ = encoder_layer.self_attn(encoder_state)
            if self.adapter_in_v and idx < max_adapter_layers:
                encoder_state = self.v_attn_adapters[idx](encoder_state)
            encoder_state = residual + encoder_state

            residual = encoder_state
            encoder_state = encoder_layer.layer_norm2(encoder_state)
            encoder_state = encoder_layer.mlp(encoder_state)
            if self.adapter_in_v and idx < max_adapter_layers:
                encoder_state = self.v_out_adapters[idx](encoder_state)
            encoder_state = residual + encoder_state

        encoder_hidden_states = encoder_hidden_states + (encoder_state,)
        return encoder_hidden_states

    def text_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.clipseg.clip.text_model.embeddings(input_ids=input_ids)
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        max_adapter_layers = len(self.l_attn_adapters) if self.adapter_in_l else 0
        num_layers = len(self.clipseg.clip.text_model.encoder.layers)

        for idx, encoder_layer in enumerate(self.clipseg.clip.text_model.encoder.layers):
            residual = hidden_states
            hidden_states = encoder_layer.layer_norm1(hidden_states)
            hidden_states, _ = encoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )
            adapter_idx = idx - (num_layers - max_adapter_layers)
            if self.adapter_in_l and adapter_idx >= 0 and adapter_idx < max_adapter_layers:
                hidden_states = self.l_attn_adapters[adapter_idx](hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = encoder_layer.layer_norm2(hidden_states)
            hidden_states = encoder_layer.mlp(hidden_states)
            if self.adapter_in_l and adapter_idx >= 0 and adapter_idx < max_adapter_layers:
                hidden_states = self.l_out_adapters[adapter_idx](hidden_states)
            hidden_states = residual + hidden_states

        last_hidden_state = self.clipseg.clip.text_model.final_layer_norm(hidden_states)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        return self.clipseg.clip.text_projection(pooled_output)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        B, C, H, W = pixel_values.shape

        vision_hidden_states = self.vision_forward(pixel_values=pixel_values)
        vision_activations = [
            vision_hidden_states[i + 1] for i in self.clipseg_config.extract_layers
        ]

        conditional_embeddings = self.text_forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if self.adapter_in_cond:
            conditional_embeddings = self.cond_adapter(conditional_embeddings)

        decoder_outputs = self.clipseg.decoder(vision_activations, conditional_embeddings)
        logits = decoder_outputs.logits.view(B, 1, H, W)

        boundary_attention = torch.sigmoid(self.boundary_enhancer(logits))
        enhanced_logits = logits + boundary_attention * logits

        return enhanced_logits


# ============================================================
#  Inference helpers
# ============================================================

def load_image(image_path: str):
    """Load and preprocess a single image. Returns tensor and original (H, W)."""
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    return transform(image).unsqueeze(0), original_size  # (1, C, H, W)


def tokenize_prompt(prompt: str, device: str):
    """Tokenize a text prompt using HuggingFace CLIPTokenizer (same vocab as OpenAI CLIP)."""
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(CLIPSEG_HF_API)
    encoded = tokenizer(
        prompt,
        max_length=CONTEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]                   # (1, context_length)
    attention_mask = encoded["attention_mask"].long()  # (1, context_length)
    return input_ids.to(device), attention_mask.to(device)


def load_model(ckpt_path: str, device: str) -> CLIPSegDenseAdapter:
    """Instantiate CLIPSegDenseAdapter and load weights from a Lightning checkpoint."""
    model = CLIPSegDenseAdapter(
        clipseg_hf_api=CLIPSEG_HF_API,
        adapter_dim=ADAPTER_DIM,
        freeze_clipseg=True,
        adapter_in_v=ADAPTER_IN_V,
        adapter_in_l=ADAPTER_IN_L,
        adapter_in_cond=ADAPTER_IN_COND,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Lightning saves weights with a "net." prefix — strip it
    cleaned = {
        (k[len("net."):] if k.startswith("net.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.to(device)
    return model


# ============================================================
#  Main
# ============================================================

def run_inference():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = "cpu"

    use_fp16 = (PRECISION == "fp16") and (device == "cuda")

    print(f"[INFO] Image     : {IMAGE_PATH}")
    print(f"[INFO] Prompt    : '{PROMPT}'")
    print(f"[INFO] Checkpoint: {CKPT_PATH}")
    print(f"[INFO] Device    : {device} | Precision: {PRECISION}")

    print("[INFO] Loading and preprocessing image...")
    image_tensor, original_size = load_image(IMAGE_PATH)
    image_tensor = image_tensor.to(device)

    print("[INFO] Tokenizing prompt...")
    input_ids, attention_mask = tokenize_prompt(PROMPT, device)

    print("[INFO] Loading model from checkpoint...")
    model = load_model(CKPT_PATH, device)

    if use_fp16:
        model = model.half()
        image_tensor = image_tensor.half()
        # input_ids and attention_mask are int tensors — do NOT cast to fp16

    print("[INFO] Running inference...")
    with torch.no_grad():
        logits = model(
            pixel_values=image_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # logits: (1, 1, H, W) → sigmoid → threshold at 0.5
        binary_mask = (torch.sigmoid(logits) > 0.5).float()

    print("[INFO] Resizing mask to original image dimensions...")
    h, w = original_size
    binary_mask_resized = TF.resize(
        binary_mask.squeeze(0).float(),     # (1, H, W)
        size=[h, w],
        interpolation=TF.InterpolationMode.NEAREST_EXACT,
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    torchvision.utils.save_image(binary_mask_resized, OUTPUT_PATH)
    print(f"[INFO] Binary mask saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()