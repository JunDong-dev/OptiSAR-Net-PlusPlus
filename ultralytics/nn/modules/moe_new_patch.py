import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTopKGate(nn.Module):
    """Gating mechanism based on cosine similarity."""

    def __init__(self, model_dim: int, num_experts: int, init_t: float = 0.5):
        super().__init__()
        proj_dim = min(model_dim // 2, 256)

        init_val = math.log(1.0 / init_t)
        self.temperature = nn.Parameter(torch.tensor([init_val]), requires_grad=True)
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.empty(proj_dim, num_experts), requires_grad=True)

        with torch.no_grad():
            nn.init.normal_(self.sim_matrix, 0, 0.01)

        self.clamp_max = math.log(1.0 / 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj_x = self.cosine_projector(x)
        norm_x = F.normalize(proj_x, p=2, dim=-1)
        norm_sim_matrix = F.normalize(self.sim_matrix, p=2, dim=0)

        logits = torch.matmul(norm_x, norm_sim_matrix)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale

        return logits


class LightweightGate(nn.Module):
    """Lightweight gating network."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        mid_dim = max(dim // 16, 8)
        self.gate = nn.Sequential(nn.Linear(dim, mid_dim), nn.GELU(), nn.Linear(mid_dim, num_experts))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x) * self.temperature


class PatchManager:
    """Utilities for patch partitioning and restoration."""

    @staticmethod
    def compute_optimal_patch_size(
        h: int, w: int, target_patches_per_dim: int = 4, min_patch: int = 4, max_patch: int = 32
    ) -> Tuple[int, int]:
        """Adaptively compute the optimal patch size."""

        def find_divisor(size, target):
            """Find the divisor of `size` closest to `target`."""
            ideal = size // target
            ideal = max(min_patch, min(max_patch, ideal))

            for delta in range(ideal):
                if ideal - delta >= 1 and size % (ideal - delta) == 0:
                    return ideal - delta
                if ideal + delta <= size and size % (ideal + delta) == 0:
                    return ideal + delta
            return 1

        patch_h = find_divisor(h, target_patches_per_dim)
        patch_w = find_divisor(w, target_patches_per_dim)

        return patch_h, patch_w

    @staticmethod
    def patchify(x: torch.Tensor, patch_h: int, patch_w: int) -> Tuple[torch.Tensor, Tuple]:
        """Partition a feature map into patches."""
        B, C, H, W = x.shape
        num_h, num_w = H // patch_h, W // patch_w

        x = x.view(B, C, num_h, patch_h, num_w, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = x.view(-1, C, patch_h, patch_w)

        return patches, (B, C, H, W, num_h, num_w, patch_h, patch_w)

    @staticmethod
    def unpatchify(patches: torch.Tensor, info: Tuple, out_channels: int = None) -> torch.Tensor:
        """Restore patches into a feature map."""
        B, C, H, W, num_h, num_w, patch_h, patch_w = info
        C_out = out_channels if out_channels else C

        x = patches.view(B, num_h, num_w, C_out, patch_h, patch_w)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C_out, H, W)

        return x


class SharedLoRAExperts(nn.Module):
    """
    Expert layer with shared weights plus LoRA adjustments.

    Design:
    1. All experts share one backbone convolution network
    2. Each expert only owns low-rank LoRA adjustment matrices
    3. LoRA adjustments are combined dynamically from the gate weights

    Structure:
        output = SharedConv(x) + Σ(gate_i × LoRA_i(x))

    where LoRA_i(x) = x @ A_i @ B_i (low-rank factorization)
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_experts: int = 4,
        rank: int = 8,
        expansion: float = 0.5,
        kernel_size: int = 3,
        use_double_lora: bool = True,  # apply LoRA after both convolutions when True
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.c_mid = int(c2 * expansion)
        self.num_experts = num_experts
        self.rank = rank
        self.use_double_lora = use_double_lora

        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(c1, self.c_mid, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(self.c_mid, c2, kernel_size, 1, kernel_size // 2, bias=False),
            nn.BatchNorm2d(c2),
        )

        self.lora1_A = nn.Parameter(torch.zeros(num_experts, self.c_mid, rank))
        self.lora1_B = nn.Parameter(torch.zeros(num_experts, rank, self.c_mid))

        if use_double_lora:
            self.lora2_A = nn.Parameter(torch.zeros(num_experts, c2, rank))
            self.lora2_B = nn.Parameter(torch.zeros(num_experts, rank, c2))

        self.lora_scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_lora_weights()

    def _init_lora_weights(self):
        """Initialize LoRA weights."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora1_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora1_B)

            if self.use_double_lora:
                nn.init.kaiming_uniform_(self.lora2_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora2_B)

    def _apply_lora_adjustment(
        self,
        x: torch.Tensor,
        gates: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LoRA adjustment.

        Args:
            x: [num_patches, C, H, W] features
            gates: [num_patches, num_experts] gate weights
            lora_A: [num_experts, C, rank] down-projection matrices
            lora_B: [num_experts, rank, C] up-projection matrices

        Returns:
            adjusted: [num_patches, C, H, W] adjusted features
        """
        N, C, H, W = x.shape

        delta_W = torch.bmm(lora_A, lora_B)  # [E, C, C]

        combined_delta = torch.einsum("ne,ecd->ncd", gates, delta_W)

        x_flat = x.view(N, C, -1)

        adjusted = torch.bmm(combined_delta, x_flat)

        adjusted = adjusted.view(N, C, H, W)

        return adjusted * self.lora_scale

    def forward(self, patches: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            patches: [num_patches, C, ph, pw] input patches
            gates: [num_patches, num_experts] gate weights

        Returns:
            output: [num_patches, C_out, ph, pw]
        """
        hidden = self.shared_conv1(patches)  # [N, c_mid, ph, pw]

        lora1_adj = self._apply_lora_adjustment(hidden, gates, self.lora1_A, self.lora1_B)
        hidden = hidden + lora1_adj

        output = self.shared_conv2(hidden)  # [N, c2, ph, pw]

        if self.use_double_lora:
            lora2_adj = self._apply_lora_adjustment(output, gates, self.lora2_A, self.lora2_B)
            output = output + lora2_adj

        return output


class PatchSharedLoRAMoE(nn.Module):
    """
    Patch-level MoE layer with shared-weight LoRA experts.

    Features:
    1. Gating decisions per patch, so different patches may pick different experts
    2. All experts share the backbone convolution; only the lightweight LoRA branches differ
    3. Efficient batched computation without invoking each expert separately
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_experts: int = 4,
        top_k: int = 2,
        rank: int = 8,
        target_patches_per_dim: int = 4,
        min_patch_size: int = 4,
        max_patch_size: int = 32,
        gating: str = "cosine",  # 'cosine' or 'linear'
        noisy_gating: bool = True,
        noise_std: float = 1.0,
        expansion: float = 0.5,
        use_double_lora: bool = True,
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.num_experts = num_experts
        self.top_k = top_k
        self.target_patches = target_patches_per_dim
        self.min_patch = min_patch_size
        self.max_patch = max_patch_size
        self.noisy_gating = noisy_gating
        self.noise_std = noise_std

        if gating == "cosine":
            self.gate = CosineTopKGate(c1, num_experts)
        else:
            self.gate = LightweightGate(c1, num_experts)

        if noisy_gating:
            self.w_noise = nn.Parameter(torch.zeros(c1, num_experts))

        self.experts = SharedLoRAExperts(
            c1=c1,
            c2=c2,
            num_experts=num_experts,
            rank=rank,
            expansion=expansion,
            use_double_lora=use_double_lora,
        )

        self._cached_patch_size = None
        self._cached_input_size = None

    def _compute_gates(
        self, gate_input: torch.Tensor, training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Top-K sparse gating (Switch-Transformer style).

        Returns:
            gates: [N, num_experts] sparse gate matrix
            router_prob: [N, num_experts] full softmax routing probabilities (differentiable)
            dispatch_mask: [N, num_experts] Top-K selection mask (non-differentiable, statistics only)
        """
        clean_logits = self.gate(gate_input)  # [N, E]

        if self.noisy_gating and training:
            raw_noise = gate_input @ self.w_noise
            noise_std = F.softplus(raw_noise) + 1e-2
            noise = torch.randn_like(clean_logits) * noise_std * self.noise_std
            logits = clean_logits + noise
        else:
            logits = clean_logits

        router_prob = F.softmax(logits, dim=-1)  # [N, E] fully differentiable

        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [N, K]

        gates = torch.zeros_like(logits)
        gates.scatter_(-1, top_k_indices, top_k_gates)  # [N, E]

        dispatch_mask = torch.zeros_like(logits)
        dispatch_mask.scatter_(-1, top_k_indices, 1.0)  # [N, E]

        return gates, router_prob, dispatch_mask

    def _compute_auxiliary_loss(self, router_prob: torch.Tensor, dispatch_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Switch-Transformer-style load-balancing loss (fully differentiable).

        L_balance = num_experts * Σ_i (f_i * P_i)

        where:
        - f_i = fraction of tokens routed to expert i (non-differentiable, used as a scalar coefficient)
        - P_i = mean routing probability of all tokens to expert i (differentiable, provides gradients)

        When all experts are used uniformly, f_i = K/E and P_i = 1/E, so
        L_balance = num_experts * E * (K/E) * (1/E) = K, which is the minimum.

        Args:
            router_prob: [N, num_experts] full softmax routing probabilities
            dispatch_mask: [N, num_experts] Top-K dispatch mask
        """
        num_experts = router_prob.shape[-1]

        f = dispatch_mask.float().mean(dim=0).detach()  # [E]

        P = router_prob.mean(dim=0)  # [E]

        balance_loss = num_experts * (f * P).sum()

        return balance_loss

    def forward(self, x: torch.Tensor, loss_coef: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] input feature map
            loss_coef: auxiliary loss coefficient

        Returns:
            output: [B, C_out, H, W]
            aux_loss: scalar loss
        """
        B, C, H, W = x.shape

        if self._cached_input_size != (H, W):
            self._cached_patch_size = PatchManager.compute_optimal_patch_size(
                H, W, self.target_patches, self.min_patch, self.max_patch
            )
            self._cached_input_size = (H, W)

        patch_h, patch_w = self._cached_patch_size

        pad_h = (patch_h - H % patch_h) % patch_h
        pad_w = (patch_w - W % patch_w) % patch_w

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        patches, info = PatchManager.patchify(x, patch_h, patch_w)

        gate_input = patches.mean(dim=(2, 3))  # [N_patches, C]

        gates, router_prob, dispatch_mask = self._compute_gates(gate_input, self.training)

        output_patches = self.experts(patches, gates)

        output = PatchManager.unpatchify(output_patches, info, self.c2)

        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :H, :W]

        aux_loss = self._compute_auxiliary_loss(router_prob, dispatch_mask) * loss_coef

        return output, aux_loss

    def get_expert_usage_stats(self) -> dict:
        """Return routing statistics for diagnostics."""
        return {
            "patch_size": self._cached_patch_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


class DualAdaptiveAttention(nn.Module):
    """Simplified dual adaptive attention module."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        ca = self.channel_attn(x).view(B, C, 1, 1)

        sa = self.spatial_attn(x)

        return x * ca * sa


class DAAM_SharedLoRA_MoE_Block(nn.Module):
    """
    DAAM + Shared-LoRA Patch-Level MoE Block

    Structure:
    x -> Norm1 -> DAA -> LayerScale -> + -> Norm2 -> MoE -> LayerScale -> + -> output
    |___________________________________|   |___________________________________|
                  residual                                residual
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        rank: int = 8,
        target_patches_per_dim: int = 4,
        min_patch_size: int = 4,
        max_patch_size: int = 32,
        use_daa: bool = True,
        gating: str = "cosine",
        noisy_gating: bool = True,
        layer_scale_init: float = 1e-2,
        loss_coef: float = 1e-2,
        expansion: float = 0.5,
        layer_scale=False,
    ):
        super().__init__()

        self.dim = dim
        self.use_daa = use_daa
        self.loss_coef = loss_coef

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        if use_daa:
            self.daa = DualAdaptiveAttention(dim)
        else:
            self.daa = nn.Identity()

        self.moe = PatchSharedLoRAMoE(
            c1=dim,
            c2=dim,
            num_experts=num_experts,
            top_k=top_k,
            rank=rank,
            target_patches_per_dim=target_patches_per_dim,
            min_patch_size=min_patch_size,
            max_patch_size=max_patch_size,
            gating=gating,
            noisy_gating=noisy_gating,
            expansion=expansion,
        )

        if layer_scale:
            self.layer_scale_daa = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
            self.layer_scale_moe = nn.Parameter(layer_scale_init * torch.ones(dim), requires_grad=True)
        else:
            self.layer_scale_daa = None
            self.layer_scale_moe = None

        self._last_moe_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.use_daa:
            attn_out = self.daa(self.norm1(x))
            if self.layer_scale_daa is not None:
                attn_out = self.layer_scale_daa.view(1, -1, 1, 1) * attn_out
            else:
                attn_out = attn_out
            x = residual + attn_out

        moe_input = self.norm2(x) if self.use_daa else self.norm1(x)
        moe_out, moe_loss = self.moe(moe_input, loss_coef=self.loss_coef)

        self._last_moe_loss = moe_loss

        if self.layer_scale_moe is not None:
            moe_out = self.layer_scale_moe.view(1, -1, 1, 1) * moe_out
        else:
            moe_out = moe_out
        output = residual + moe_out

        return output

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_last_moe_loss":
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result


class PLoRA_MoE(nn.Module):
    """
    C3k2 with Patch-Level Shared-LoRA DAAM-MoE

    Features:
    1. CSPNet-style split structure
    2. Patch-level MoE decisions
    3. Efficient expert design with shared weights + LoRA
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        num_experts: int = 4,
        top_k: int = 2,
        rank: int = 8,
        target_patches_per_dim: int = 4,
        loss_coef: float = 1e-2,
        use_daa: bool = True,
        layer_scale=False,
        min_patch_size: int = 1,
        max_patch_size: int = 32,
        gating: str = "cosine",
    ):
        super().__init__()

        self.c = int(c2 * e)  # hidden channels

        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, 0, bias=False), nn.BatchNorm2d(2 * self.c), nn.SiLU(inplace=True)
        )

        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, 1, 0, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True)
        )

        self.m = nn.ModuleList(
            [
                DAAM_SharedLoRA_MoE_Block(
                    dim=self.c,
                    num_experts=num_experts,
                    top_k=top_k,
                    rank=rank,
                    target_patches_per_dim=target_patches_per_dim,
                    min_patch_size=min_patch_size,
                    max_patch_size=max_patch_size,
                    use_daa=use_daa,
                    gating=gating,
                    loss_coef=loss_coef,
                    layer_scale=layer_scale,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))

        for block in self.m:
            y.append(block(y[-1]))

        return self.cv2(torch.cat(y, 1))

    def get_moe_loss(self) -> torch.Tensor:
        """Collect the MoE auxiliary losses of all blocks."""
        total_loss = 0.0
        count = 0

        for block in self.m:
            loss = getattr(block, "_last_moe_loss", None)
            if loss is not None:
                total_loss = total_loss + loss
                count += 1

        return total_loss if count > 0 else torch.tensor(0.0)
