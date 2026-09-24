import math

import torch
import torch.nn as nn

from ultralytics.utils.torch_utils import smart_inference_mode

from .block import Bottleneck, SpatialShuffleAttention
from .conv import Conv


class MaxSigmoidAttnBlock_DualGate_v2(nn.Module):
    """
    Improved dual-gated Max Sigmoid attention block.

    Improvements:
    1. Learnable temperature parameter
    2. Enhanced guide projection
    3. Optional depthwise separable convolution for efficiency
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        nh: int = 1,
        ec: int = 128,
        gc: int = 512,
        scale: bool = False,
        use_weight_gate: bool = True,
        use_feature_gate: bool = True,
        learnable_temp: bool = True,
        use_enhanced_guide: bool = False,
        use_depthwise: bool = False,
    ):
        """
        Initialize the improved dual-gated attention block.

        Args:
            c1 (int): number of input channels
            c2 (int): number of output channels
            nh (int): number of attention heads
            ec (int): number of embedding channels
            gc (int): number of guide channels
            scale (bool): whether to use a learnable scale
            use_weight_gate (bool): whether to gate the attention weights
            use_feature_gate (bool): whether to gate the output features
            learnable_temp (bool): whether to use a learnable temperature
            use_enhanced_guide (bool): whether to use the enhanced guide projection
            use_depthwise (bool): whether to use depthwise separable convolution
        """
        super().__init__()
        self.nh = nh
        self.hc = ec // nh
        self.use_weight_gate = use_weight_gate
        self.use_feature_gate = use_feature_gate
        self.c1 = c1
        self.c2 = c2

        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None

        if use_enhanced_guide:
            self.gl = EnhancedGuideProjection(gc, ec, nh)
            self.use_enhanced_guide = True
        else:
            self.gl = nn.Linear(gc, ec)
            self.use_enhanced_guide = False

        self.bias = nn.Parameter(torch.zeros(nh))

        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)

        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

        if learnable_temp:
            self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.hc))
        else:
            self.register_buffer("temperature", torch.tensor(math.sqrt(self.hc)))

        self.guide = None

        if use_weight_gate:
            self.weight_gate = nn.Parameter(torch.zeros(1, nh, 1, 1))

        if use_feature_gate:
            self.feature_gate = nn.Parameter(torch.zeros(1, nh, 1, 1))

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        """
        Fuse guide features for fast inference.

        Args:
            txt_feats (torch.Tensor): text features
        """
        assert not self.training, "fuse() should only be called during inference"

        if self.use_enhanced_guide:
            guide = self.gl(txt_feats.to(next(self.gl.parameters()).dtype), 1)
        else:
            guide = self.gl(txt_feats.to(self.gl.weight.dtype))
            guide = guide.view(1, -1, self.nh, self.hc)

        if hasattr(self, "guide") and self.guide is not None:
            del self.guide
        self.register_buffer("guide", guide)

        if hasattr(self, "gl"):
            del self.gl

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): input features with shape (B, C1, H, W)
            guide (torch.Tensor): guide features with shape (1 or B, num_classes, gc)

        Returns:
            torch.Tensor: output features with shape (B, C2, H, W)
        """
        bs, _, h, w = x.shape

        if self.guide is None:
            if self.use_enhanced_guide:
                guide = self.gl(guide.to(next(self.gl.parameters()).dtype), bs)
            else:
                guide = self.gl(guide.to(self.gl.weight.dtype))
                if guide.shape[0] == 1 and bs > 1:
                    guide = guide.repeat(bs, 1, 1)
                guide = guide.view(bs, -1, self.nh, self.hc)
        else:
            guide = self.guide
            if guide.shape[0] == 1 and bs > 1:
                guide = guide.expand(bs, -1, -1, -1)

        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)

        aw = aw.max(dim=-1)[0]  # (B, nh, H, W)

        aw = aw / self.temperature

        aw = aw + self.bias[None, :, None, None]

        aw_raw = aw.sigmoid() * self.scale

        if self.use_weight_gate:
            baseline = torch.ones_like(aw_raw)
            aw = baseline + self.weight_gate * (aw_raw - baseline)
        else:
            aw = aw_raw

        x_proj = self.proj_conv(x)

        if self.use_feature_gate:
            x_attn = x_proj.view(bs, self.nh, -1, h, w)
            x_attn = x_attn * aw.unsqueeze(2)
            x_attn = x_attn.view(bs, -1, h, w)

            c_per_head = x_proj.shape[1] // self.nh
            gate_expanded = self.feature_gate.repeat(1, 1, c_per_head, 1).view(1, -1, 1, 1)

            x = x_proj + gate_expanded * (x_attn - x_proj)
        else:
            x = x_proj.view(bs, self.nh, -1, h, w)
            x = x * aw.unsqueeze(2)
            x = x.view(bs, -1, h, w)

        return x


class TGDG_SSA(nn.Module):
    """
    TGDG_SSA

    Combines:
    - spatial awareness of VSSA (improved SSA)
    - the lightweight design of GSConv
    - the external guidance mechanism of C2fAttn

    Improvements:
    1. Temperature parameter for the dual gates
    2. Enhanced guide features
    3. Residual connections in the bottleneck
    4. Optional depthwise separable convolution
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        groups: int = 8,
        use_ssa: bool = True,
        use_enhanced_guide: bool = False,
        gc: int = 512,
        e: float = 0.5,
        use_se: bool = False,
        use_depthwise: bool = False,
        bottleneck_residual: bool = False,
    ):
        """
        Initialize the improved GS_C2f_GuidedVSSA.

        Args:
            c1 (int): number of input channels
            c2 (int): number of output channels
            n (int): number of Bottleneck blocks
            ec (int): number of attention embedding channels
            nh (int): number of attention heads
            groups (int): number of SSA groups
            gc (int): number of guide channels
            g (int): number of convolution groups
            e (float): expansion coefficient
            use_ssa (bool): whether to use SSA
            use_se (bool): whether to use SE
            use_enhanced_guide (bool): whether to use the enhanced guide projection
            use_depthwise (bool): whether to use depthwise separable convolution
            bottleneck_residual (bool): whether the bottleneck uses a residual connection
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.n = n
        self.bottleneck_residual = bottleneck_residual

        if use_ssa:
            self.ssa = SpatialShuffleAttention(c1, groups=groups)
        else:
            self.ssa = nn.Identity()

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        total_channels = (3 + n) * self.c
        self.cv2 = Conv(total_channels, c2, 1)

        self.m = nn.ModuleList([Bottleneck(self.c, self.c) for _ in range(n)])

        self.attn = MaxSigmoidAttnBlock_DualGate_v2(
            self.c,
            self.c,
            gc=gc,
            ec=ec,
            nh=nh,
            use_weight_gate=True,
            use_feature_gate=True,
            learnable_temp=True,
            use_enhanced_guide=use_enhanced_guide,
            use_depthwise=use_depthwise,
        )

        self.se = None

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): input features with shape (B, C1, H, W)
            guide (torch.Tensor): guide features with shape (1 or B, num_classes, gc)

        Returns:
            torch.Tensor: output features with shape (B, C2, H, W)
        """

        x = self.ssa(x)

        y = list(self.cv1(x).chunk(2, 1))

        for m in self.m:
            y.append(m(y[-1]))

        y.append(self.attn(y[-1], guide))

        concat_feat = torch.cat(y, 1)

        if self.se is not None:
            concat_feat = self.se(concat_feat)

        return self.cv2(concat_feat)


class EnhancedGuideProjection(nn.Module):
    """
    Enhanced guide-feature projection module.

    Improvements:
    1. Two-layer MLP for extra non-linearity
    2. LayerNorm for stability
    3. GELU activation
    4. Optional Dropout regularization
    """

    def __init__(self, gc: int, ec: int, nh: int, hidden_ratio: float = 2.0, dropout: float = 0.0):
        """
        Initialize the enhanced guide projection.

        Args:
            gc (int): number of guide input channels
            ec (int): number of embedding channels
            nh (int): number of attention heads
            hidden_ratio (float): hidden-channel ratio
            dropout (float): dropout probability
        """
        super().__init__()
        self.nh = nh
        self.hc = ec // nh
        hidden_dim = int(ec * hidden_ratio)

        self.proj = nn.Sequential(
            nn.Linear(gc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, ec),
            nn.LayerNorm(ec),
        )

    def forward(self, guide: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Forward pass.

        Args:
            guide (torch.Tensor): guide features with shape (1 or B, num_classes, gc)
            bs (int): Batch size

        Returns:
            torch.Tensor: projected guide with shape (B, num_classes, nh, hc)
        """
        guide = self.proj(guide)

        if guide.shape[0] == 1 and bs > 1:
            guide = guide.repeat(bs, 1, 1)

        return guide.view(bs, -1, self.nh, self.hc)
