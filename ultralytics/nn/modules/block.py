# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import logging
from typing import Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ultralytics.utils.torch_utils import smart_inference_mode

from .conv import Conv, DWConv, GhostConv

__all__ = (
    "DFL",
    "Proto",
    "SPPF",
    "C2f",
    "C3",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "ResNetBlock",
    "ResNetLayer",
    "ContrastiveHead",
    "BNContrastiveHead",
    "AConv",
    "ADown",
    "C3k2",
    "C3k",
    "Attention",
    "PSABlock",
    "PSA",
    "C2PSA",
    "C2fPSA",
    "AAttn",
    "ABlock",
    "EnhancedConvolutionalBlock",
    "DualAdaptiveAttention",
    "DAAM",
    "TopkRouting",
    "KVGather",
    "BiLevelRoutingDeformableAttention",
    "BSPPF",
    "C2fAttn",
    "MaxSigmoidAttnBlock",
    "GSConv",
    "GSBottleneck",
    "SpatialShuffleAttention",
    "VSSA",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x, w):
        return x

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func

        USE_FLASH_ATTN = True
        logger.warning("FlashAttention is available！！！")
    else:
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")


class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(q.contiguous().half(), k.contiguous().half(), v.contiguous().half()).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = v @ attn.transpose(-2, -1)

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)

    Notes:
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class EnhancedConvolutionalBlock(nn.Module):
    """
    Enhanced Convolutional Block (ECB) that applies two 1x1 convolutions with optional residual connection.
    """

    def __init__(self, input_channels, hidden_channels=None, output_channels=None, dropout_rate=0.0, use_residual=True):
        super().__init__()

        # If not specified, keep the number of channels constant
        output_channels = output_channels or input_channels
        hidden_channels = hidden_channels or input_channels

        # First 1x1 convolution layer
        self.conv_reduce = Conv(input_channels, hidden_channels, k=1)

        # Second 1x1 convolution layer
        self.conv_expand = Conv(hidden_channels, output_channels, k=1)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Flag to control residual connection
        self.use_residual = use_residual

    def forward(self, x):
        # Store the input for potential residual connection
        residual = x

        # First convolution
        x = self.conv_reduce(x)
        x = self.dropout(x)

        # Second convolution
        x = self.conv_expand(x)
        x = self.dropout(x)

        # Add residual connection if enabled
        if self.use_residual:
            return x.contiguous() + residual.contiguous()
        else:
            return x.contiguous()


class DualAdaptiveAttention(nn.Module):
    """
    Dual Adaptive Attention (DAA).
    """

    def __init__(self, dim):
        super().__init__()

        # Input projection
        self.input_proj = Conv(dim, dim, k=1)
        self.activation = nn.GELU()

        # Output projection
        self.output_proj = Conv(dim, dim, k=1)

        # Depth-wise convolutions for local and global feature extraction
        self.local_conv = Conv(dim, dim, k=3, p=1, g=dim)
        self.global_conv = Conv(dim, dim, k=3, p=3, g=dim, d=3)

        # Channel reduction for attention computation
        self.channel_reducer_local = Conv(dim, dim // 2, k=1)
        self.channel_reducer_global = Conv(dim, dim // 2, k=1)

        # Attention squeeze operation
        self.attention_squeeze = Conv(2, 2, k=7, p=3)

        # Final channel mixing
        self.channel_mixer = Conv(dim // 2, dim, k=1)

    def forward(self, x):
        # Store input for residual connection
        residual = x.clone()

        # Input projection and activation
        x = self.input_proj(x)
        x = self.activation(x)

        # Local feature extraction
        local_features = self.local_conv(x)

        # Global feature extraction
        global_features = self.global_conv(local_features)

        # Compute attention for local and global features
        attn_local = self.channel_reducer_local(local_features)
        attn_global = self.channel_reducer_global(global_features)

        # Concatenate local and global attention
        attn_combined = torch.cat([attn_local, attn_global], dim=1)

        # Compute average and max attention
        attn_avg = torch.mean(attn_combined, dim=1, keepdim=True)
        attn_max, _ = torch.max(attn_combined, dim=1, keepdim=True)

        # Aggregate average and max attention
        attn_pooled = torch.cat([attn_avg, attn_max], dim=1)

        # Apply attention squeeze and sigmoid activation
        attn_weights = self.attention_squeeze(attn_pooled).sigmoid()

        # Compute weighted attention
        weighted_attn = attn_local * attn_weights[:, 0, :, :].unsqueeze(1) + attn_global * attn_weights[
            :, 1, :, :
        ].unsqueeze(1)

        # Mix channels in the attention
        attn_mixed = self.channel_mixer(weighted_attn)

        # Apply attention to global features
        x = global_features * attn_mixed

        # Output projection
        x = self.output_proj(x)

        # Add residual connection
        x = x + residual

        return x.contiguous()


class DAAM(nn.Module):
    """
    Dual Attention Adaptive Module (DAAM) that combines attention and ECB
    with optional layer scaling.
    """

    def __init__(self, dim, use_auto_layer_scaling=True, layer_scale_init_value=1e-2):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.daa = DualAdaptiveAttention(dim)
        self.ecb = EnhancedConvolutionalBlock(dim, dim)

        self.use_auto_layer_scaling = use_auto_layer_scaling
        if use_auto_layer_scaling:
            self.layer_scale_daa = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_ecb = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_daa = None
            self.layer_scale_ecb = None

    def forward(self, x):
        # Apply attention
        attention_output = self.daa(self.norm1(x))
        if self.use_auto_layer_scaling:
            attention_output = self.layer_scale_daa.unsqueeze(-1).unsqueeze(-1) * attention_output
        x = x + attention_output

        # Apply ECB
        ecb_output = self.ecb(self.norm2(x))
        if self.use_auto_layer_scaling:
            ecb_output = self.layer_scale_ecb.unsqueeze(-1).unsqueeze(-1) * ecb_output
        x = x + ecb_output

        return x


class TopkRouting(nn.Module):
    """
    Differentiable top-k routing with scaling, adapted from bi-level routing attention.
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim**-0.5
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """

        query, key = query.detach(), key.detach()
        attn_logit = (query * self.scale) @ key.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    """
    KVGather module for efficient key-value pair selection based on routing indices.
    This module is part of the bi-level routing attention mechanism.
    """

    def __init__(self):
        super().__init__()

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            # (n, p^2, p^2, w^2, c_kv) without mem cpy
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv),
            # (n, p^2, k, w^2, c_kv)
        )

        return topk_kv


class BiLevelRoutingDeformableAttention(nn.Module):
    """
    Bi-Level Routing Deformable Attention module.

    This module combines local attention with global routing and deformable convolutions
    for enhanced feature extraction in computer vision tasks.
    """

    def __init__(
        self,
        dim,
        num_windows=7,
        num_heads=2,
        qk_dim=None,
        qk_scale=None,
        kv_per_window=2,
        kv_downsample_mode="ada_maxpool",
        topk=4,
        side_conv=3,
        use_deformable=False,
        off_conv=7,
    ):
        """
        Initialize the Bi-Level Routing Deformable Attention.

        Args:
            dim (int): Number of input channels.
            num_windows (int): Number of windows in each dimension for local attention.
            num_heads (int): Number of attention heads.
            qk_dim (int): Dimension of query and key vectors. If None, set to dim. Default is None.
            qk_scale (float): Scaling factor for query-key dot product. If None, set to 1/sqrt(qk_dim).
            kv_per_window (int): Number of key-value pairs per window for downsampling.
            kv_downsample_mode (str): Mode for downsampling key-value pairs. Options: 'ada_avgpool', 'ada_maxpool'.
            topk (int): Number of top attention scores to consider in routing.
            side_dwconv (int): Kernel size for depthwise convolution in LEPE. Set to 0 to disable.
            use_deformable (bool):
            auto_pad (bool): Whether to automatically pad input to match window size.
        """
        super().__init__()
        self.dim = dim
        self.num_windows = num_windows
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, (
            "qk_dim and dim must be divisible by num_heads!"
        )
        self.scale = qk_scale or self.qk_dim**-0.5

        # Side-enhanced convolution (SEC)
        self.sec = (
            nn.Conv2d(dim, dim, kernel_size=side_conv, stride=1, padding=side_conv // 2, groups=dim)
            if side_conv > 0
            else lambda x: torch.zeros_like(x)
        )

        # Global routing settings
        self.topk = topk
        self.router = TopkRouting(qk_dim=self.qk_dim, qk_scale=self.scale, topk=self.topk)
        self.kv_gather = KVGather()

        # Query, Key, Value projections
        self.query_proj = Conv(dim, dim, 1)
        self.kv_proj = Conv(dim, dim * 2, 1)

        # Key-Value downsampling
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_window = kv_per_window
        if self.kv_downsample_mode == "ada_avgpool":
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_window)
        elif self.kv_downsample_mode == "ada_maxpool":
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_window)
        else:
            self.kv_down = nn.Identity()

        self.attn_act = nn.Softmax(dim=-1)
        self.use_deformable = use_deformable
        self.off_conv = off_conv

        # Offset prediction
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(
                self.dim, self.dim, kernel_size=self.off_conv, stride=1, padding=self.off_conv // 2, groups=self.dim
            ),
            nn.BatchNorm2d(dim),
            nn.Conv2d(self.dim, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )

    @torch.no_grad()
    def _get_reference_points(self, height, width, batch_size, dtype, device):
        """
        Generate reference points for deformable attention.

        Args:
            height (int): Height of the feature map.
            width (int): Width of the feature map.
            batch_size (int): Batch size.
            dtype (torch.dtype): Data type of the tensor.
            device (torch.device): Device to create the tensor on.

        Returns:
            torch.Tensor: Reference points of shape (batch_size, height, width, 2).
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
            torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(width - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(height - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(batch_size, -1, -1, -1)
        return ref

    def forward(self, x):
        x = rearrange(x, "n c h w -> n h w c")

        # Auto-padding
        batch_size, height_in, width_in, channels = x.size()
        pad_left = pad_top = 0
        pad_right = (self.num_windows - width_in % self.num_windows) % self.num_windows
        pad_bottom = (self.num_windows - height_in % self.num_windows) % self.num_windows
        x = F.pad(x, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, height, width, _ = x.size()  # padded size

        # Reshape input for window-based processing
        x = rearrange(x, "n (j h) (i w) c -> n c (j h) (i w)", j=self.num_windows, i=self.num_windows)

        # Query projection
        query = self.query_proj(x)
        query_offset = query
        query = rearrange(query, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)

        # Deformable offset calculation
        offset = self.offset_predictor(query_offset).contiguous()
        height_key, width_key = offset.size(2), offset.size(3)
        offset = einops.rearrange(offset, "b p h w -> b h w p")
        dtype, device = x.dtype, x.device
        batch_size, _, _, _ = offset.size()
        reference = self._get_reference_points(height_key, width_key, batch_size, dtype, device)

        pos = (offset + reference).clamp(-0.05, +0.05)

        # Apply deformable sampling
        if self.use_deformable:
            x_sampled = F.grid_sample(
                input=x,
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode="bilinear",
                align_corners=True,
            )
        else:
            x_sampled = x

        # Key-Value projection
        kv = self.kv_proj(x_sampled)
        kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.num_windows, i=self.num_windows)

        # Reshape for pixel-wise and window-wise operations
        query_pixel = rearrange(query, "n p2 h w c -> n p2 (h w) c")
        kv_pixel = rearrange(kv, "n p2 h w c -> (n p2) c h w")

        # Downsample key-value pairs
        kv_pixel = self.kv_down(kv_pixel)
        kv_pixel = rearrange(kv_pixel, "(n j i) c h w -> n (j i) (h w) c", j=self.num_windows, i=self.num_windows)

        # Window-wise query and key
        query_window, key_window = query.mean([2, 3]), kv[..., 0 : self.qk_dim].mean([2, 3])

        # Side-enhanced convolution (SEC)
        sec = self.sec(
            rearrange(
                kv[..., self.qk_dim :], "n (j i) h w c -> n c (j h) (i w)", j=self.num_windows, i=self.num_windows
            ).contiguous()
        )
        sec = rearrange(sec, "n c (j h) (i w) -> n (j h) (i w) c", j=self.num_windows, i=self.num_windows)

        # Global routing
        routing_weights, routing_indices = self.router(query_window, key_window)

        # Gather key-value pairs based on routing
        kv_pixel_selected = self.kv_gather(r_idx=routing_indices, r_weight=routing_weights, kv=kv_pixel)
        key_pixel_selected, value_pixel_selected = kv_pixel_selected.split([self.qk_dim, self.dim], dim=-1)

        # Reshape for multi-head attention
        key_pixel_selected = rearrange(key_pixel_selected, "n p2 k w2 (m c) -> (n p2) m c (k w2)", m=self.num_heads)
        value_pixel_selected = rearrange(value_pixel_selected, "n p2 k w2 (m c) -> (n p2) m (k w2) c", m=self.num_heads)
        query_pixel = rearrange(query_pixel, "n p2 w2 (m c) -> (n p2) m w2 c", m=self.num_heads)

        # Compute attention weights and apply attention
        attn_weights = (query_pixel * self.scale) @ key_pixel_selected
        attn_weights = self.attn_act(attn_weights)
        out = attn_weights @ value_pixel_selected

        # Reshape output and add SEC
        out = rearrange(
            out,
            "(n j i) m (h w) c -> n (j h) (i w) (m c)",
            j=self.num_windows,
            i=self.num_windows,
            h=height // self.num_windows,
            w=width // self.num_windows,
        )
        out = out + sec

        # Remove padding if applied
        out = out[:, :height_in, :width_in, :].contiguous()

        return rearrange(out, "n h w c -> n c h w").contiguous()


class BSPPF(nn.Module):
    # The Bi-Level Routing Deformable Spatial Pyramid Pooling - Fast (BSPPF) layer
    # is used to dynamically adjust the allocation of multi-scale feature space.

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.brda = BiLevelRoutingDeformableAttention(c_)

    def forward(self, x):
        x = self.cv1(x)

        # Apply Bi-Level Routing Deformable Attention with residual connection
        x = x + self.brda(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)).contiguous()


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, c3k=False, shortcut=False, gc=512, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class MaxSigmoidAttnBlock(nn.Module):
    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = ec // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
        self.guide = None

    @smart_inference_mode()
    def fuse(self, txt_feats):
        assert not self.training
        guide = self.gl(txt_feats.to(self.gl.weight.dtype))
        guide = guide.view(1, -1, self.nh, self.hc)
        del self.guide
        self.register_buffer("guide", guide)
        del self.gl

    def forward(self, x, guide):
        bs, _, h, w = x.shape

        if self.guide is None:
            guide = self.gl(guide.to(self.gl.weight.dtype))

            if guide.shape[0] == 1 and bs > 1:
                guide = guide.repeat(bs, 1, 1)
            guide = guide.view(bs, -1, self.nh, self.hc)
        else:
            guide = self.guide

        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        x = x.view(bs, -1, h, w)

        return x


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1).contiguous()


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(GSConv(c1, c_, 3, 1), GSConv(c_, c2, 3, 1, act=False))

    def forward(self, x):
        return self.conv_lighting(x) + x


class SpatialShuffleAttention(nn.Module):
    """
    Original Spatial Shuffle Attention (SSA), kept for backward compatibility.
    """

    def __init__(self, dim, groups=8, dropout_rate=0.01):
        super().__init__()
        self.groups = groups
        self.dim = dim

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.weight_max = nn.Parameter(torch.zeros(1, dim // (2 * groups), 1, 1))
        self.bias_max = nn.Parameter(torch.ones(1, dim // (2 * groups), 1, 1))
        self.weight_avg = nn.Parameter(torch.zeros(1, dim // (2 * groups), 1, 1))
        self.bias_avg = nn.Parameter(torch.ones(1, dim // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, channels, height, width = x.shape
        channels_per_group = channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        x = self.channel_shuffle(x, 2)
        x_1, x_2 = x.chunk(2, dim=1)
        avg_pool = self.avg_pool(x_1)
        max_pool = self.max_pool(x_2)
        avg_attention = self.weight_avg * avg_pool + self.bias_avg
        max_attention = self.weight_max * max_pool + self.bias_max
        channel_attention = torch.cat((max_attention, avg_attention), dim=1)
        channel_attention = self.sigmoid(channel_attention)
        x = x * self.dropout(channel_attention)
        out = x.contiguous().view(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out.contiguous()


class VSSA(nn.Module):
    """
    VoVGSCSP module with Spatial Shuffle Attention (VSSA).
    """

    def __init__(self, in_channels, out_channels, num_gsb=4, expansion_factor=0.5, dropout_rate=0.05, groups=8):
        """
        Initialize the VSSA module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_gsb (int): Number of GSBottleneck layers. Default is 4.
            expansion_factor (float): Factor to determine the number of hidden channels. Default is 0.5.
        """
        super().__init__()
        # Lazy import to avoid circular imports
        from .c2fatt import SpatialShuffleAttention

        hidden_channels = int(out_channels * expansion_factor)  # Calculate hidden channels
        self.conv_input_1 = Conv(in_channels, hidden_channels, k=1, s=1)
        self.conv_input_2 = Conv(in_channels, hidden_channels, k=1, s=1)

        # GSBottleneck sequence
        self.gsb_sequence = nn.Sequential(
            *(GSBottleneck(hidden_channels, hidden_channels, e=1.0) for _ in range(num_gsb))
        )
        self.conv_output = Conv(2 * hidden_channels, out_channels, k=1)

        # Spatial Shuffle Attention
        self.spatial_shuffle_attention = SpatialShuffleAttention(in_channels, groups=groups)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply Spatial Shuffle Attention and dropout
        attended_features = self.dropout(self.spatial_shuffle_attention(x))

        # Process through GSBottleneck sequence
        gsb_output = self.gsb_sequence(self.conv_input_1(attended_features))

        # Direct path through second input convolution
        direct_path = self.conv_input_2(attended_features)

        # Concatenate and process through output convolution
        combined_features = torch.cat((direct_path, gsb_output), dim=1)
        output = self.conv_output(combined_features)

        return output.contiguous()
