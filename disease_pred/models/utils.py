from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn


def ConvUnit(in_channels, out_channels, kernel_size, stride, padding) -> nn.Sequential:
    """Convolutional unit with batch norm and ReLU activation

    Args:
        in_channels (_type_): _description_
        out_channels (_type_): _description_
        kernel_size (_type_): _description_
        stride (_type_): _description_
        padding (_type_): _description_

    Returns:
        nn.Sequential: _description_
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


def ConvTransposeUnit(in_channels, out_channels, kernel_size, stride, padding, output_padding) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


def attention_rollout(
    attentions: List[torch.Tensor], discard_ratio: float = 0.9, head_fusion: str = "max"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs attention rollout.

    adapted from https://jacobgil.github.io/deeplearning/vision-transformer-explainability

    Args:
        attentions (List[torch.Tensor]): List of attention maps.
        discard_ratio (float, optional): Discard ratio. Defaults to 0.9.
        head_fusion (str, optional): Head fusion method. Available are "mean", "max", and "min".
            Defaults to "max".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Joint attention map and layer-wise maps.
    """
    joint_attention = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    layer_attention = []
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            identity = torch.eye(attention_heads_fused.size(-1), device=attention.device)
            a = (attention_heads_fused + 1.0 * identity) / 2
            a = a / a.sum(dim=-1)

            layer_attention.append(a)

            joint_attention = torch.matmul(a, joint_attention)

    # Look at the total attention between the class token,
    # and the image patches
    joint_mask = joint_attention[0, 0, 1:]
    layer_mask = torch.stack([a[0, 0, 1:] for a in layer_attention]).to(attentions[0].device)
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(joint_mask.size(-1) ** 0.5)
    joint_mask = joint_mask.reshape(width, width)
    layer_mask = layer_mask.reshape(len(attentions), width, width)

    joint_mask = joint_mask.cpu().numpy()
    layer_mask = layer_mask.cpu().numpy()
    joint_mask = joint_mask / np.max(joint_mask)
    layer_mask = layer_mask / np.max(layer_mask, axis=(1, 2))[:, None, None]

    return joint_mask, layer_mask


def pick_target_subset(targets: Dict[str, Any], choice: np.array) -> Dict[str, Any]:
    picked = {}
    for k, v in targets.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                picked[k] = [v_[choice] for v_ in v]
            else:
                picked[k] = [v[c] for c in choice]
        else:
            picked[k] = v[choice]
    return picked
