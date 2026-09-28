from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorPair = Tuple[torch.Tensor, torch.Tensor]
CDAMOutput = Union[TensorPair, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for [B, C, H, W] tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def _canonical_backbone_type(backbone_type: str) -> str:
    backbone_type = str(backbone_type).lower()
    if "resnet" in backbone_type:
        return "resnet"
    if "mobilenet" in backbone_type:
        return "mobilenet"
    if "efficientnet" in backbone_type:
        return "efficientnet"
    if "hrnet" in backbone_type:
        return "hrnet"
    if "convnext" in backbone_type:
        return "convnext"
    if "swin" in backbone_type:
        return "swin"
    if "mambaout" in backbone_type:
        return "mambaout"
    if "transformer" in backbone_type:
        return "transformer"
    return backbone_type or "cnn"


def _resolve_norm_type(norm_type: str, backbone_type: str) -> str:
    norm_type = str(norm_type).lower()
    if norm_type != "auto":
        return norm_type

    backbone_type = _canonical_backbone_type(backbone_type)
    if backbone_type in {"resnet", "cnn"}:
        return "bn"
    if backbone_type in {"mobilenet", "efficientnet", "hrnet"}:
        return "gn"
    if backbone_type in {"convnext", "swin", "mambaout", "transformer"}:
        return "ln2d"
    return "gn"


def _best_group_count(num_channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def build_norm(
    num_channels: int,
    norm_type: str = "auto",
    backbone_type: str = "cnn",
    avoid_batchnorm: bool = False,
    eps: float = 1e-6,
) -> nn.Module:
    norm_type = _resolve_norm_type(norm_type, backbone_type)
    if avoid_batchnorm and norm_type == "bn":
        norm_type = "gn"

    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "gn":
        return nn.GroupNorm(_best_group_count(num_channels), num_channels)
    if norm_type == "ln2d":
        return LayerNorm2d(num_channels, eps=eps)
    if norm_type == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}.")


def build_context_norm(
    num_channels: int,
    norm_type: str = "auto",
    backbone_type: str = "cnn",
    eps: float = 1e-6,
) -> nn.Module:
    """Normalization for [B, C, 1, 1] context tensors."""
    resolved_norm_type = _resolve_norm_type(norm_type, backbone_type)
    if resolved_norm_type == "ln2d":
        return LayerNorm2d(num_channels, eps=eps)
    if resolved_norm_type == "identity":
        return nn.Identity()
    return nn.GroupNorm(1, num_channels)


def build_act(act_type: str = "silu") -> nn.Module:
    act_type = str(act_type).lower()
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "silu":
        return nn.SiLU(inplace=True)
    if act_type == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown act_type: {act_type}.")


class ConvNormAct(nn.Module):
    """Conv2d -> adaptive norm -> activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_type: str = "auto",
        act_type: str = "silu",
        backbone_type: str = "cnn",
        avoid_batchnorm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            build_norm(
                out_channels,
                norm_type=norm_type,
                backbone_type=backbone_type,
                avoid_batchnorm=avoid_batchnorm,
                eps=eps,
            ),
            build_act(act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 -> norm/act -> pointwise 1x1 -> norm/act."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "auto",
        act_type: str = "silu",
        backbone_type: str = "cnn",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                norm_type=norm_type,
                act_type=act_type,
                backbone_type=backbone_type,
                eps=eps,
            ),
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                norm_type=norm_type,
                act_type=act_type,
                backbone_type=backbone_type,
                eps=eps,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CDAMBase(nn.Module):
    """Shared implementation for CDAM encoder modules.

    CDAM builds a change-aware modulation map from two complementary signals:
    1. a local bi-temporal change prior |F1 - F2|,
    2. a global domain/style statistic difference based on mean and standard
       deviation.

    The same modulation map is applied to both temporal branches so that
    change-related responses are enhanced while domain/style disturbance can be
    suppressed.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        reduction: int = 4,
        min_hidden_channels: int = 16,
        max_hidden_channels: Optional[int] = None,
        modulation_mode: str = "residual",
        use_suppression: bool = True,
        use_spatial_refine: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_aux_loss: bool = False,
        lambda_sparsity: float = 0.01,
        lambda_domain: float = 0.1,
        residual_scale_init: float = 1.0,
        learnable_residual_scale: bool = False,
        output_refine_type: str = "full",
        identity_init: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if min_hidden_channels <= 0:
            raise ValueError("min_hidden_channels must be positive.")
        if max_hidden_channels is not None and max_hidden_channels <= 0:
            raise ValueError("max_hidden_channels must be positive when set.")
        if modulation_mode not in {"residual", "suppressive"}:
            raise ValueError(
                f"Unknown modulation_mode: {modulation_mode}. "
                "Expected 'residual' or 'suppressive'."
            )
        if output_refine_type not in {"lightweight", "full"}:
            raise ValueError(
                f"Unknown output_refine_type: {output_refine_type}. "
                "Expected 'lightweight' or 'full'."
            )

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = max(in_channels // reduction, min_hidden_channels)
            if max_hidden_channels is not None:
                hidden_channels = min(hidden_channels, max_hidden_channels)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        self.hidden_channels = hidden_channels
        self.reduction = reduction
        self.min_hidden_channels = min_hidden_channels
        self.max_hidden_channels = max_hidden_channels
        self.modulation_mode = modulation_mode
        self.use_suppression = use_suppression
        self.use_spatial_refine = use_spatial_refine
        self.dropout = dropout
        self.eps = eps
        self.use_aux_loss = use_aux_loss
        self.lambda_sparsity = lambda_sparsity
        self.lambda_domain = lambda_domain
        self.output_refine_type = output_refine_type
        self.identity_init = identity_init
        if learnable_residual_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_buffer("residual_scale", torch.tensor(float(residual_scale_init)))

        self.diff_encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
        )

        # GroupNorm is used on the [B, hidden, 1, 1] domain branch so training
        # remains valid for batch size 1.
        self.domain_encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.attention_generator = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels * 2,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(self.hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        if use_spatial_refine:
            self.spatial_refine = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.spatial_refine = None

        if output_refine_type == "full":
            self.output_refine = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # Keep the residual adapter lightweight and channel-adaptive. A full
            # 3x3 C->C convolution grows quadratically with backbone width
            # (e.g. ResNet50/ConvNeXt-Base) and can dominate the pretrained encoder.
            self.output_refine = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channels, in_channels, kernel_size=1, bias=True),
            )

        self._init_weights()
        if identity_init:
            # Start as an identity mapping for every backbone. The attention
            # branch is still supervised by the auxiliary loss, while the
            # residual adapter learns how strongly to affect pretrained
            # features during training.
            nn.init.zeros_(self.output_refine[-1].weight)
            if self.output_refine[-1].bias is not None:
                nn.init.zeros_(self.output_refine[-1].bias)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_stats(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute domain/style statistic difference in shape [B, C, 1, 1]."""
        dtype = f1.dtype
        f1_stats = f1.float()
        f2_stats = f2.float()

        mu1 = f1_stats.mean(dim=(2, 3), keepdim=True)
        mu2 = f2_stats.mean(dim=(2, 3), keepdim=True)

        var1 = ((f1_stats - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
        var2 = ((f2_stats - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
        std1 = torch.sqrt(var1 + self.eps)
        std2 = torch.sqrt(var2 + self.eps)

        domain_stat = torch.abs(mu1 - mu2) + torch.abs(std1 - std2)
        return domain_stat.to(dtype=dtype)

    def _prepare_gt(
        self,
        gt: torch.Tensor,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            if gt.shape[1] > 1:
                gt = gt.argmax(dim=1, keepdim=True)
        else:
            raise ValueError("gt must have shape [B,H,W], [B,1,H,W], or one-hot [B,C,H,W].")

        gt = gt.to(device=device, dtype=torch.float32)
        if gt.shape[-2:] != target_size:
            gt = F.interpolate(gt, size=target_size, mode="nearest")
        return (gt > 0.5).float() if gt.max() <= 1.0 else (gt > 0.0).float()

    def _bce_from_prob(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(prob.float().clamp(self.eps, 1.0 - self.eps))
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _compute_aux_loss(
        self,
        gt: Optional[torch.Tensor],
        change_attention: torch.Tensor,
        diff: torch.Tensor,
        domain_stat: torch.Tensor,
        spatial_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del diff, spatial_weight

        attn_mean = change_attention.float().mean(dim=1, keepdim=True)
        if gt is None:
            zero = attn_mean.sum() * 0.0
            return {
                "loss_cdam": zero,
                "loss_align": zero,
                "loss_sparsity": zero,
                "loss_domain": zero,
            }

        gt = self._prepare_gt(
            gt=gt,
            target_size=change_attention.shape[-2:],
            device=change_attention.device,
        )
        loss_align = self._bce_from_prob(attn_mean, gt)

        loss_sparsity = attn_mean.mean()
        domain_strength = domain_stat.float().mean(dim=1, keepdim=True)
        domain_strength = domain_strength.expand_as(attn_mean)
        loss_domain = ((1.0 - gt) * domain_strength * attn_mean).mean()
        loss_cdam = (
            loss_align
            + self.lambda_sparsity * loss_sparsity
            + self.lambda_domain * loss_domain
        )

        return {
            "loss_cdam": loss_cdam,
            "loss_align": loss_align,
            "loss_sparsity": loss_sparsity,
            "loss_domain": loss_domain,
        }

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> CDAMOutput:
        if f1.dim() != 4 or f2.dim() != 4:
            raise ValueError("f1 and f2 must both have shape [B,C,H,W].")
        if f1.shape != f2.shape:
            raise ValueError(f"f1 and f2 must have identical shapes, got {f1.shape} and {f2.shape}.")
        if f1.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, but received {f1.shape[1]}."
            )

        b, _, h, w = f1.shape

        diff = torch.abs(f1 - f2)
        diff_feat = self.diff_encoder(diff)

        domain_stat = self._compute_stats(f1, f2)
        domain_gate = self.domain_encoder(domain_stat)
        domain_map = domain_gate.expand(-1, -1, h, w)

        fuse = torch.cat([diff_feat, domain_map], dim=1)
        change_attention = self.attention_generator(fuse)

        if self.spatial_refine is not None:
            spatial_weight = self.spatial_refine(diff)
            change_attention = change_attention * spatial_weight
        else:
            spatial_weight = torch.ones((b, 1, h, w), device=f1.device, dtype=f1.dtype)

        if self.use_suppression:
            suppression_attention = 1.0 - change_attention
        else:
            suppression_attention = torch.zeros_like(change_attention)

        if self.modulation_mode == "residual":
            raw_out1 = f1 + f1 * change_attention
            raw_out2 = f2 + f2 * change_attention
        elif self.modulation_mode == "suppressive":
            raw_out1 = f1 + f1 * change_attention - f1 * suppression_attention
            raw_out2 = f2 + f2 * change_attention - f2 * suppression_attention
        else:
            raise ValueError(f"Unknown modulation_mode: {self.modulation_mode}")

        residual_scale = self.residual_scale.to(dtype=f1.dtype)
        out1 = f1 + residual_scale * self.output_refine(raw_out1 - f1)
        out2 = f2 + residual_scale * self.output_refine(raw_out2 - f2)

        if self.use_aux_loss:
            aux_dict = self._compute_aux_loss(
                gt=gt,
                change_attention=change_attention,
                diff=diff,
                domain_stat=domain_stat,
                spatial_weight=spatial_weight,
            )
            aux_dict.update(
                {
                    "change_prior": diff.detach(),
                    "domain_stat": domain_stat,
                    "change_attention": change_attention,
                    "suppression_attention": suppression_attention.detach(),
                    "spatial_weight": spatial_weight.detach(),
                }
            )
            return out1, out2, aux_dict

        return out1, out2


class CDAMv1(_CDAMBase):
    """CDAM v1: original full residual adapter version."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_hidden_channels"] = None
        kwargs["output_refine_type"] = "full"
        kwargs["identity_init"] = False
        kwargs["learnable_residual_scale"] = False
        super().__init__(*args, **kwargs)


class CDAMv2(_CDAMBase):
    """CDAM v2: channel-capped lightweight residual adapter version."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["max_hidden_channels"] = 128
        kwargs["output_refine_type"] = "lightweight"
        kwargs["identity_init"] = True
        kwargs["learnable_residual_scale"] = True
        super().__init__(*args, **kwargs)


class CDAMv4(_CDAMBase):
    """CDAM v4: full-hidden ablation version with hidden_channels equal to C."""

    def __init__(self, *args, **kwargs) -> None:
        in_channels = kwargs.get("in_channels", args[0] if args else None)
        if in_channels is None:
            raise ValueError("CDAMv4 requires in_channels.")
        kwargs["hidden_channels"] = int(in_channels)
        kwargs["max_hidden_channels"] = None
        kwargs["output_refine_type"] = "full"
        kwargs["identity_init"] = False
        kwargs["learnable_residual_scale"] = False
        super().__init__(*args, **kwargs)


# Backward-compatible alias for older imports. New code should use CDAMv1, CDAMv2, CDAMv3, or CDAMv4.
CDAM = CDAMv1


class CDAMv3(nn.Module):
    """CDAM v3: backbone-adaptive, softly initialized change-aware modulation.

    This version is intended for cross-backbone comparison. It avoids fixed
    ResNet18 assumptions by reading each stage's real channel count from the
    caller, using adaptive normalization/convolution choices, and applying a
    small learnable gamma so the module starts close to identity.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        reduction: int = 4,
        min_hidden: int = 32,
        max_hidden: int = 256,
        backbone_type: str = "cnn",
        stage_index: int = 0,
        norm_type: str = "auto",
        act_type: str = "silu",
        conv_type: str = "auto",
        modulation_mode: str = "residual",
        use_domain_branch: bool = True,
        use_spatial_branch: bool = True,
        use_channel_branch: bool = True,
        use_suppression: bool = False,
        init_gamma: float = 0.1,
        dropout: float = 0.0,
        eps: float = 1e-6,
        use_aux_loss: bool = False,
        lambda_spatial: float = 0.5,
        lambda_sparsity: float = 0.01,
        lambda_domain: float = 0.1,
        lambda_channel: float = 0.01,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if min_hidden <= 0 or max_hidden <= 0:
            raise ValueError("min_hidden and max_hidden must be positive.")
        if min_hidden > max_hidden:
            raise ValueError("min_hidden must be <= max_hidden.")

        self.in_channels = in_channels
        self.backbone_type = _canonical_backbone_type(backbone_type)
        self.stage_index = int(stage_index)
        self.norm_type = str(norm_type).lower()
        self.act_type = str(act_type).lower()
        self.conv_type = self._resolve_conv_type(conv_type)
        self.modulation_mode = str(modulation_mode).lower()
        if self.modulation_mode not in {"residual", "suppressive"}:
            raise ValueError(
                f"Unknown modulation_mode: {modulation_mode}. "
                "Expected 'residual' or 'suppressive'."
            )

        if hidden_channels is None:
            hidden_channels = max(in_channels // reduction, min_hidden)
            hidden_channels = min(hidden_channels, max_hidden)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        self.hidden_channels = hidden_channels
        self.reduction = reduction
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
        self.use_domain_branch = use_domain_branch
        self.use_spatial_branch = use_spatial_branch
        self.use_channel_branch = use_channel_branch
        self.use_suppression = use_suppression
        self.dropout = dropout
        self.eps = eps
        self.use_aux_loss = use_aux_loss
        self.lambda_spatial = lambda_spatial
        self.lambda_sparsity = lambda_sparsity
        self.lambda_domain = lambda_domain
        self.lambda_channel = lambda_channel
        self.stage_scale = min(0.5 + 0.25 * self.stage_index, 1.5)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

        conv_block = self._make_conv_block
        self.diff_encoder = nn.Sequential(
            conv_block(in_channels, hidden_channels),
            nn.Dropout2d(dropout),
            conv_block(hidden_channels, hidden_channels),
        )

        self.domain_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            build_context_norm(
                hidden_channels,
                norm_type=self.norm_type,
                backbone_type=self.backbone_type,
                eps=eps,
            ),
            build_act(self.act_type),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=True),
            build_act(self.act_type),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.spatial_branch = nn.Sequential(
            ConvNormAct(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                backbone_type=self.backbone_type,
                eps=eps,
            ),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.attention_generator = nn.Sequential(
            ConvNormAct(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                backbone_type=self.backbone_type,
                eps=eps,
            ),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.output_refine = self._make_conv_block(in_channels, in_channels)
        self.beta_generator = nn.Sequential(
            ConvNormAct(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                norm_type=self.norm_type,
                act_type=self.act_type,
                backbone_type=self.backbone_type,
                eps=eps,
            ),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
        )

        self._init_weights()
        self._init_stable_outputs()

    def _resolve_conv_type(self, conv_type: str) -> str:
        conv_type = str(conv_type).lower()
        if conv_type != "auto":
            if conv_type not in {"standard", "depthwise_separable"}:
                raise ValueError(f"Unknown conv_type: {conv_type}.")
            return conv_type
        if self.backbone_type in {"mobilenet", "efficientnet"}:
            return "depthwise_separable"
        return "standard"

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.conv_type == "depthwise_separable":
            return DepthwiseSeparableConv(
                in_channels,
                out_channels,
                norm_type=self.norm_type,
                act_type=self.act_type,
                backbone_type=self.backbone_type,
                eps=self.eps,
            )
        return ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
            backbone_type=self.backbone_type,
            eps=self.eps,
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
                if isinstance(module, LayerNorm2d):
                    nn.init.ones_(module.norm.weight)
                    nn.init.zeros_(module.norm.bias)
                else:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_stable_outputs(self) -> None:
        attention_conv = self.attention_generator[-2]
        if isinstance(attention_conv, nn.Conv2d) and attention_conv.bias is not None:
            nn.init.constant_(attention_conv.bias, -2.0)

        spatial_conv = self.spatial_branch[-2]
        if isinstance(spatial_conv, nn.Conv2d) and spatial_conv.bias is not None:
            nn.init.constant_(spatial_conv.bias, -1.0)

        beta_conv = self.beta_generator[-1]
        if isinstance(beta_conv, nn.Conv2d):
            nn.init.zeros_(beta_conv.weight)
            if beta_conv.bias is not None:
                nn.init.zeros_(beta_conv.bias)

    def _compute_stats(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
    ) -> torch.Tensor:
        dtype = f1.dtype
        f1_stats = f1.float()
        f2_stats = f2.float()
        mu1 = f1_stats.mean(dim=(2, 3), keepdim=True)
        mu2 = f2_stats.mean(dim=(2, 3), keepdim=True)
        var1 = ((f1_stats - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
        var2 = ((f2_stats - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
        std1 = torch.sqrt(var1 + self.eps)
        std2 = torch.sqrt(var2 + self.eps)
        return (torch.abs(mu1 - mu2) + torch.abs(std1 - std2)).to(dtype=dtype)

    def _prepare_gt(
        self,
        gt: torch.Tensor,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            if gt.shape[1] > 1:
                gt = gt.argmax(dim=1, keepdim=True)
        else:
            raise ValueError("gt must have shape [B,H,W], [B,1,H,W], or one-hot [B,C,H,W].")

        gt = gt.to(device=device, dtype=torch.float32)
        if gt.shape[-2:] != target_size:
            gt = F.interpolate(gt, size=target_size, mode="nearest")
        return (gt > 0.5).float() if gt.max() <= 1.0 else (gt > 0.0).float()

    def _bce_from_prob(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(prob.float().clamp(self.eps, 1.0 - self.eps))
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _compute_aux_loss(
        self,
        gt: Optional[torch.Tensor],
        change_attention: torch.Tensor,
        spatial_weight: torch.Tensor,
        channel_gate: torch.Tensor,
        domain_stat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        attn_mean = change_attention.float().mean(dim=1, keepdim=True)
        if gt is None:
            zero = attn_mean.sum() * 0.0
            return {
                "loss_cdam": zero,
                "loss_align": zero,
                "loss_spatial": zero,
                "loss_sparsity": zero,
                "loss_domain": zero,
                "loss_channel": zero,
            }

        gt = self._prepare_gt(gt, change_attention.shape[-2:], change_attention.device)
        loss_align = self._bce_from_prob(attn_mean, gt)
        if self.use_spatial_branch:
            loss_spatial = self._bce_from_prob(spatial_weight, gt)
        else:
            loss_spatial = loss_align.sum() * 0.0
        loss_sparsity = attn_mean.mean()

        if self.use_domain_branch:
            domain_strength = domain_stat.float().mean(dim=1, keepdim=True)
            domain_strength = domain_strength.expand_as(attn_mean)
            loss_domain = ((1.0 - gt) * domain_strength * attn_mean).mean()
        else:
            loss_domain = loss_align.sum() * 0.0

        if self.use_channel_branch:
            loss_channel = ((channel_gate.float().mean(dim=1, keepdim=True) - 0.5) ** 2).mean()
        else:
            loss_channel = loss_align.sum() * 0.0

        loss_cdam = (
            loss_align
            + self.lambda_spatial * loss_spatial
            + self.lambda_sparsity * loss_sparsity
            + self.lambda_domain * loss_domain
            + self.lambda_channel * loss_channel
        )
        return {
            "loss_cdam": loss_cdam,
            "loss_align": loss_align,
            "loss_spatial": loss_spatial,
            "loss_sparsity": loss_sparsity,
            "loss_domain": loss_domain,
            "loss_channel": loss_channel,
        }

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> CDAMOutput:
        if f1.dim() != 4 or f2.dim() != 4:
            raise ValueError("f1 and f2 must both have shape [B,C,H,W].")
        if f1.shape != f2.shape:
            raise ValueError(f"f1 and f2 must have identical shapes, got {f1.shape} and {f2.shape}.")
        if f1.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, but received {f1.shape[1]}.")

        b, _, h, w = f1.shape
        diff = torch.abs(f1 - f2)
        diff_feat = self.diff_encoder(diff)

        domain_stat = self._compute_stats(f1, f2)
        domain_stat_norm = domain_stat / (domain_stat.mean(dim=1, keepdim=True) + self.eps)
        if self.use_domain_branch:
            domain_input = torch.cat([domain_stat, domain_stat_norm], dim=1)
            domain_gate = self.domain_encoder(domain_input)
            domain_map = domain_gate.expand(-1, -1, h, w)
        else:
            domain_map = torch.zeros_like(diff_feat)

        if self.use_channel_branch:
            pooled_diff = diff.mean(dim=(2, 3), keepdim=True)
            channel_input = torch.cat([pooled_diff, domain_stat], dim=1)
            channel_gate = self.channel_mlp(channel_input)
        else:
            channel_gate = torch.ones((b, self.in_channels, 1, 1), device=f1.device, dtype=f1.dtype)

        if self.use_spatial_branch:
            spatial_weight = self.spatial_branch(diff)
        else:
            spatial_weight = torch.ones((b, 1, h, w), device=f1.device, dtype=f1.dtype)

        fuse = torch.cat([diff_feat, domain_map], dim=1)
        attention_base = self.attention_generator(fuse)
        change_attention = (attention_base * channel_gate * spatial_weight).clamp(0.0, 1.0)

        gamma = self.gamma.to(dtype=f1.dtype) * self.stage_scale
        if self.modulation_mode == "residual":
            delta1 = self.output_refine(f1 * change_attention)
            delta2 = self.output_refine(f2 * change_attention)
            out1 = f1 + gamma * delta1
            out2 = f2 + gamma * delta2
        elif self.modulation_mode == "suppressive":
            suppression_attention = 1.0 - change_attention
            delta1 = f1 * change_attention - f1 * suppression_attention
            delta2 = f2 * change_attention - f2 * suppression_attention
            out1 = f1 + gamma * self.output_refine(delta1)
            out2 = f2 + gamma * self.output_refine(delta2)
        else:
            raise ValueError(f"Unknown modulation_mode: {self.modulation_mode}.")

        if self.use_suppression:
            suppression_attention = 1.0 - change_attention
        else:
            suppression_attention = torch.zeros_like(change_attention)

        if self.use_aux_loss:
            aux_dict = self._compute_aux_loss(
                gt=gt,
                change_attention=change_attention,
                spatial_weight=spatial_weight,
                channel_gate=channel_gate,
                domain_stat=domain_stat,
            )
            aux_dict.update(
                {
                    "change_attention": change_attention,
                    "spatial_weight": spatial_weight.detach(),
                    "channel_gate": channel_gate.detach(),
                    "domain_stat": domain_stat,
                    "suppression_attention": suppression_attention.detach(),
                    "gamma": gamma.detach().reshape(()),
                }
            )
            return out1, out2, aux_dict

        return out1, out2


class CDAMv3Stack(nn.Module):
    """Apply CDAMv3 to a list of encoder stages.

    channels_list must come from the actual backbone feature tensors or
    feature_info; do not hard-code ResNet18's [64, 128, 256, 512].
    """

    def __init__(
        self,
        channels_list: List[int],
        backbone_type: str = "cnn",
        norm_type: str = "auto",
        conv_type: str = "auto",
        modulation_mode: str = "residual",
        use_aux_loss: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.modules_list = nn.ModuleList(
            CDAMv3(
                in_channels=channels,
                backbone_type=backbone_type,
                stage_index=stage_index,
                norm_type=norm_type,
                conv_type=conv_type,
                modulation_mode=modulation_mode,
                use_aux_loss=use_aux_loss,
                **kwargs,
            )
            for stage_index, channels in enumerate(channels_list)
        )

    def forward(
        self,
        feats1: List[torch.Tensor],
        feats2: List[torch.Tensor],
        gt: Optional[torch.Tensor] = None,
    ):
        if len(feats1) != len(feats2) or len(feats1) != len(self.modules_list):
            raise ValueError("Feature list lengths must match CDAMv3Stack stages.")

        outs1 = []
        outs2 = []
        aux_items = []
        for stage_index, (module, feat1, feat2) in enumerate(zip(self.modules_list, feats1, feats2)):
            result = module(feat1, feat2, gt=gt)
            if len(result) == 3:
                out1, out2, aux = result
                aux_items.append((stage_index, aux))
            else:
                out1, out2 = result
            outs1.append(out1)
            outs2.append(out2)

        if not self.use_aux_loss:
            return outs1, outs2

        aux_total: Dict[str, torch.Tensor] = {}
        loss_keys = (
            "loss_cdam",
            "loss_align",
            "loss_spatial",
            "loss_sparsity",
            "loss_domain",
            "loss_channel",
        )
        for key in loss_keys:
            values = [aux[key] for _, aux in aux_items if key in aux]
            if values:
                aux_total[key] = torch.stack(values).mean()
        for stage_index, aux in aux_items:
            for key in ("change_attention", "spatial_weight", "channel_gate", "domain_stat"):
                if key in aux:
                    aux_total[f"stage_{stage_index}_{key}"] = aux[key]
        return outs1, outs2, aux_total


if __name__ == "__main__":
    B, C, H, W = 2, 128, 32, 32
    f1 = torch.randn(B, C, H, W)
    f2 = torch.randn(B, C, H, W)
    gt = torch.randint(0, 2, (B, 1, H, W)).float()

    for module_cls in (CDAMv1, CDAMv2, CDAMv4):
        module = module_cls(
            in_channels=C,
            hidden_channels=64,
            modulation_mode="suppressive",
            use_suppression=True,
            use_spatial_refine=True,
            dropout=0.1,
            use_aux_loss=True,
        )

        out1, out2, aux = module(f1, f2, gt)

        print(module_cls.__name__)
        print("out1:", out1.shape)
        print("out2:", out2.shape)
        print("loss_cdam:", aux["loss_cdam"])
        print("change_attention:", aux["change_attention"].shape)
        print("domain_stat:", aux["domain_stat"].shape)

    module = CDAMv3(
        in_channels=C,
        hidden_channels=64,
        backbone_type="resnet",
        modulation_mode="residual",
        dropout=0.1,
        use_aux_loss=True,
    )
    out1, out2, aux = module(f1, f2, gt)

    print("CDAMv3")
    print("out1:", out1.shape)
    print("out2:", out2.shape)
    print("loss_cdam:", aux["loss_cdam"])
    print("change_attention:", aux["change_attention"].shape)
    print("domain_stat:", aux["domain_stat"].shape)
