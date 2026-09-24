# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    BSPPF,
    C2PSA,
    C3,
    DAAM,  # GS_C2f_GuidedVSSA #, C3k2_DAAM_MoE
    DFL,
    PSA,
    SPPF,
    VSSA,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fPSA,
    C3k2,
    ContrastiveHead,
    GhostBottleneck,
    Proto,
    ResNetLayer,
)

# from .moe_old_patch import C3k2_DAAM_MoE
from .c2fatt import TGDG_SSA
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import Detect, LRPCHead, OptiSARNetPlusPlusDetect
from .moe_new_patch import PLoRA_MoE
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "SPPF",
    "C3",
    "C2f",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "TransformerEncoderLayer",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OptiSARNetPlusPlusDetect",
    "LRPCHead",
    "ContrastiveHead",
    "BNContrastiveHead",
    "AConv",
    "ADown",
    "Attention",
    "PSA",
    "DAAM",
    "BSPPF",
    "VSSA",
    "TGDG_SSA",
    "PLoRA_MoE",
)
