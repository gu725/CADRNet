from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCDAMPriorFusion(nn.Module):
    """Fuse four CDAM change/style priors into decoder-channel guidance."""

    def __init__(
        self,
        in_channels_list: Sequence[int],
        decoder_channels: int,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(in_channels_list) != 4:
            raise ValueError("MultiScaleCDAMPriorFusion expects exactly four CDAM levels.")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")

        self.in_channels_list = [int(ch) for ch in in_channels_list]
        if any(ch <= 0 for ch in self.in_channels_list):
            raise ValueError("all in_channels_list entries must be positive.")
        self.decoder_channels = int(decoder_channels)
        if hidden_channels is None:
            hidden_channels = max(self.decoder_channels // 4, 16)
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        self.hidden_channels = int(hidden_channels)

        self.attention_projs = nn.ModuleList(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=True)
            for in_channels in self.in_channels_list
        )
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 4, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.hidden_channels, self.decoder_channels, kernel_size=1, bias=True),
        )

        self.style_projs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=False),
            )
            for in_channels in self.in_channels_list
        )
        self.style_fusion = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 4, self.hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.hidden_channels, self.decoder_channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _check_level(
        self,
        values: Sequence[torch.Tensor],
        name: str,
    ) -> None:
        if len(values) != 4:
            raise ValueError(f"{name} must contain exactly four tensors.")
        for idx, (tensor, channels) in enumerate(zip(values, self.in_channels_list)):
            if tensor is None:
                raise ValueError(f"{name}[{idx}] is None.")
            if tensor.dim() != 4:
                raise ValueError(f"{name}[{idx}] must have shape [B,C,H,W].")
            if tensor.shape[1] != channels:
                raise ValueError(
                    f"{name}[{idx}] expected {channels} channels, got {tensor.shape[1]}."
                )

    def forward(
        self,
        attention_maps: Sequence[torch.Tensor],
        style_stats: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_level(attention_maps, "attention_maps")
        self._check_level(style_stats, "style_stats")

        aligned_attention = []
        for attention, proj in zip(attention_maps, self.attention_projs):
            attention_proj = proj(attention)
            if attention_proj.shape[-2:] != target_size:
                attention_proj = F.interpolate(
                    attention_proj,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_attention.append(attention_proj)
        attention_cat = torch.cat(aligned_attention, dim=1)
        attention_multi = self.attention_fusion(attention_cat)

        style_features = []
        for style, proj in zip(style_stats, self.style_projs):
            if style.shape[-2:] != (1, 1):
                style = F.adaptive_avg_pool2d(style, output_size=1)
            style_features.append(proj(style))
        style_cat = torch.cat(style_features, dim=1)
        style_multi = self.style_fusion(style_cat)

        return attention_multi, style_multi
