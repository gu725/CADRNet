import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import List, Optional, Tuple


class ExchangeType(str, Enum):
    LAYER = 'le'
    RAND_LAYER = 'rle'
    CHANNEL = 'ce'
    RAND_CHANNEL = 'rce'
    SPATIAL = 'se'
    RAND_SPATIAL = 'rse'
    FDSE = 'fdse'


class _CDAMGuidedExchange(nn.Module):
    """Shared validation and prior preparation for trainable exchanges."""

    def __init__(self, in_channels_list: List[int], eps: float = 1e-6) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("in_channels_list must contain at least one feature level.")
        if any(int(channels) <= 0 for channels in in_channels_list):
            raise ValueError("all entries in in_channels_list must be positive.")
        self.in_channels_list = [int(channels) for channels in in_channels_list]
        self.eps = float(eps)

    def _validate_inputs(
        self,
        feat_a: List[torch.Tensor],
        feat_b: List[torch.Tensor],
    ) -> None:
        if len(feat_a) != len(feat_b):
            raise ValueError("featA and featB must have the same number of feature levels.")
        if len(feat_a) != len(self.in_channels_list):
            raise ValueError(
                f"Expected {len(self.in_channels_list)} feature levels, but received {len(feat_a)}."
            )
        for index, (feature_a, feature_b, channels) in enumerate(
            zip(feat_a, feat_b, self.in_channels_list)
        ):
            if feature_a.shape != feature_b.shape:
                raise ValueError(
                    f"Paired features must have identical shapes at level {index}, "
                    f"got {feature_a.shape} and {feature_b.shape}."
                )
            if feature_a.ndim != 4 or feature_a.shape[1] != channels:
                raise ValueError(
                    f"Level {index} expected a BCHW tensor with {channels} channels, "
                    f"but received {feature_a.shape}."
                )

    def _layer_ids(self, layers: Optional[List[int]]) -> set:
        layer_ids = set(range(len(self.in_channels_list))) if layers is None else set(layers)
        invalid = sorted(index for index in layer_ids if index < 0 or index >= len(self.in_channels_list))
        if invalid:
            raise ValueError(f"Invalid exchange layer indices: {invalid}.")
        return layer_ids

    @staticmethod
    def _select_guide(
        guides: Optional[List[Optional[torch.Tensor]]],
        index: int,
    ) -> Optional[torch.Tensor]:
        if guides is None or index >= len(guides):
            return None
        return guides[index]

    def _resize_guide(self, guide: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        guide = guide.to(device=feature.device, dtype=feature.dtype)
        if guide.shape[-2:] != feature.shape[-2:]:
            guide = F.interpolate(
                guide,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return guide

    def _attention_channels(
        self,
        guide: Optional[torch.Tensor],
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        if guide is None:
            difference = torch.abs(feature_a - feature_b).mean(dim=(2, 3), keepdim=True)
            reference = difference.mean(dim=1, keepdim=True)
            return difference / (difference + reference + self.eps)
        guide = self._resize_guide(guide, feature_a)
        if guide.shape[1] == 1:
            guide = guide.expand(-1, feature_a.shape[1], -1, -1)
        elif guide.shape[1] != feature_a.shape[1]:
            guide = guide.mean(dim=1, keepdim=True).expand(-1, feature_a.shape[1], -1, -1)
        return guide.mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)

    def _style_stat(
        self,
        style_stat: Optional[torch.Tensor],
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        if style_stat is None:
            work_a = feature_a.float()
            work_b = feature_b.float()
            mean_a = work_a.mean(dim=(2, 3), keepdim=True)
            mean_b = work_b.mean(dim=(2, 3), keepdim=True)
            std_a = torch.sqrt(work_a.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
            std_b = torch.sqrt(work_b.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
            style_stat = torch.abs(mean_a - mean_b) + torch.abs(std_a - std_b)
            return style_stat.to(dtype=feature_a.dtype)

        style_stat = style_stat.to(device=feature_a.device, dtype=feature_a.dtype)
        if style_stat.shape[-2:] != (1, 1):
            style_stat = F.adaptive_avg_pool2d(style_stat, output_size=1)
        if style_stat.shape[1] == 1:
            style_stat = style_stat.expand(-1, feature_a.shape[1], -1, -1)
        elif style_stat.shape[1] != feature_a.shape[1]:
            style_stat = style_stat.mean(dim=1, keepdim=True).expand(
                -1, feature_a.shape[1], -1, -1
            )
        return style_stat


class _ChannelGuidanceGate(nn.Module):
    def __init__(self, channels: int, reduction: int, min_hidden_channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, min_hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)


class FrequencyDecoupledSpectralExchange(_CDAMGuidedExchange):
    """Share low-frequency amplitude while retaining phase and fine boundaries."""

    def __init__(
        self,
        in_channels_list: List[int],
        reduction: int = 4,
        min_hidden_channels: int = 16,
        init_strength: float = 0.1,
        init_cutoff: float = 0.2,
        min_cutoff: float = 0.05,
        max_cutoff: float = 0.5,
        mask_temperature: float = 0.02,
    ) -> None:
        super().__init__(in_channels_list)
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        if not 0.0 < init_strength < 1.0:
            raise ValueError("init_strength must be in (0, 1).")
        if not 0.0 < min_cutoff < init_cutoff < max_cutoff <= 0.5:
            raise ValueError("cutoffs must satisfy 0 < min < init < max <= 0.5.")
        if mask_temperature <= 0.0:
            raise ValueError("mask_temperature must be positive.")
        self.gates = nn.ModuleList(
            [
                _ChannelGuidanceGate(channels, reduction, min_hidden_channels)
                for channels in self.in_channels_list
            ]
        )
        initial_strength_logit = math.log(init_strength / (1.0 - init_strength))
        cutoff_ratio = (init_cutoff - min_cutoff) / (max_cutoff - min_cutoff)
        initial_cutoff_logit = math.log(cutoff_ratio / (1.0 - cutoff_ratio))
        self.strength_logits = nn.Parameter(
            torch.full((len(self.in_channels_list),), initial_strength_logit)
        )
        self.cutoff_logits = nn.Parameter(
            torch.full((len(self.in_channels_list),), initial_cutoff_logit)
        )
        self.min_cutoff = float(min_cutoff)
        self.max_cutoff = float(max_cutoff)
        self.mask_temperature = float(mask_temperature)

    def _low_frequency_mask(
        self,
        height: int,
        width: int,
        index: int,
        device: torch.device,
    ) -> torch.Tensor:
        frequency_y = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(height, 1)
        frequency_x = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(1, -1)
        radius = torch.sqrt(frequency_y.square() + frequency_x.square())
        cutoff = self.min_cutoff + (self.max_cutoff - self.min_cutoff) * torch.sigmoid(
            self.cutoff_logits[index]
        )
        mask = torch.sigmoid((cutoff.float() - radius) / self.mask_temperature)
        return mask.view(1, 1, height, width // 2 + 1)

    def forward(
        self,
        featA: List[torch.Tensor],
        featB: List[torch.Tensor],
        attention_maps: Optional[List[Optional[torch.Tensor]]] = None,
        style_stats: Optional[List[Optional[torch.Tensor]]] = None,
        layers: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._validate_inputs(featA, featB)
        layer_ids = self._layer_ids(layers)
        out_a = list(featA)
        out_b = list(featB)

        for index, (feature_a, feature_b) in enumerate(zip(featA, featB)):
            if index not in layer_ids:
                continue
            attention = self._attention_channels(
                self._select_guide(attention_maps, index), feature_a, feature_b
            )
            style = self._style_stat(
                self._select_guide(style_stats, index), feature_a, feature_b
            )
            style_reference = style.mean(dim=1, keepdim=True)
            normalized_style = style / (style_reference + self.eps)
            style_confidence = style / (style + style_reference + self.eps)
            commonness = 1.0 - attention
            learned_gate = torch.sigmoid(
                self.gates[index].net(torch.cat([torch.tanh(normalized_style), commonness], dim=1))
            )
            strength = torch.sigmoid(self.strength_logits[index]).view(1, 1, 1, 1)
            channel_weight = strength * learned_gate * style_confidence * commonness

            work_a = feature_a.float()
            work_b = feature_b.float()
            spectrum_a = torch.fft.rfft2(work_a, norm="ortho")
            spectrum_b = torch.fft.rfft2(work_b, norm="ortho")
            amplitude_a = torch.abs(spectrum_a)
            amplitude_b = torch.abs(spectrum_b)
            consensus_amplitude = 0.5 * (amplitude_a + amplitude_b)
            frequency_mask = self._low_frequency_mask(
                feature_a.shape[-2], feature_a.shape[-1], index, feature_a.device
            )
            exchange_weight = channel_weight.float() * frequency_mask
            target_amplitude_a = amplitude_a + exchange_weight * (consensus_amplitude - amplitude_a)
            target_amplitude_b = amplitude_b + exchange_weight * (consensus_amplitude - amplitude_b)
            phase_a = spectrum_a / amplitude_a.clamp_min(self.eps)
            phase_b = spectrum_b / amplitude_b.clamp_min(self.eps)
            exchanged_a = torch.fft.irfft2(
                phase_a * target_amplitude_a,
                s=feature_a.shape[-2:],
                norm="ortho",
            )
            exchanged_b = torch.fft.irfft2(
                phase_b * target_amplitude_b,
                s=feature_b.shape[-2:],
                norm="ortho",
            )
            out_a[index] = exchanged_a.to(dtype=feature_a.dtype)
            out_b[index] = exchanged_b.to(dtype=feature_b.dtype)

        return out_a, out_b


class FeatureExchanger:
    """
    支持多种特征交换操作：
      - LAYER/RAND_LAYER: 对整个特征列表进行层级交换
      - CHANNEL/RAND_CHANNEL: 对列表中指定张量的通道做交换
      - SPATIAL/RAND_SPATIAL: 对列表中指定张量的空间维度做交换
      - FDSE: 可训练的 CDAM 引导交换，由 CADRNet 持有参数并调用
    训练时随机交换，推理时固定交换。

    参数说明：
      thresh: 随机层交换的概率阈值
      p: 通道/空间交换的间隔或概率参数
      layers: 可选层索引列表，指定对哪些层(feature)执行 channel 或 spatial 交换，
              为 None 时默认对所有层进行操作。
    """

    def __init__(self, training: bool = True):
        self.training = training

    @staticmethod
    def layer_exchange(
        x: List[torch.Tensor],
        y: List[torch.Tensor],
        p: int = 2,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for i in range(0, len(x), p):
            x[i], y[i] = y[i], x[i]
        return x, y

    @staticmethod
    def random_layer_exchange(
        x: List[torch.Tensor],
        y: List[torch.Tensor],
        thresh: float = 0.5,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        for i in range(len(x)):
            if torch.rand(1).item() < thresh:
                x[i], y[i] = y[i], x[i]
        return x, y

    @staticmethod
    def channel_exchange(
        x1: torch.Tensor,
        x2: torch.Tensor,
        p: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, C, _, _ = x1.shape
        mask = (torch.arange(C, device=x1.device) % p == 0).view(1, C, 1, 1)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    @staticmethod
    def random_channel_exchange(
        x1: torch.Tensor,
        x2: torch.Tensor,
        thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, C, _, _ = x1.shape
        mask = (torch.rand(C, device=x1.device) < thresh).view(1, C, 1, 1)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    @staticmethod
    def spatial_exchange(
        x1: torch.Tensor,
        x2: torch.Tensor,
        p: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, W = x1.shape
        mask = (torch.arange(W, device=x1.device) % p == 0).view(1, 1, 1, W)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    @staticmethod
    def random_spatial_exchange(
        x1: torch.Tensor,
        x2: torch.Tensor,
        thresh: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, _, W = x1.shape
        mask = (torch.rand(W, device=x1.device) < thresh).view(1, 1, 1, W)
        out1 = torch.where(mask, x2, x1)
        out2 = torch.where(mask, x1, x2)
        return out1, out2

    def exchange(
        self,
        featA: List[torch.Tensor],
        featB: List[torch.Tensor],
        mode: ExchangeType = ExchangeType.LAYER,
        thresh: float = 0.5,
        p: int = 2,
        layers: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        执行指定模式的交换操作，支持层级、通道和空间三种交换。

        参数：
          featA, featB: 待交换的特征列表
          mode: 交换模式
          thresh: 随机层交换阈值
          p: 通道/空间交换参数 (步长或概率)
          layers: 层索引列表，None 则对所有层操作
        """
        if mode == ExchangeType.LAYER:
            return self.layer_exchange(featA, featB, p)

        if mode == ExchangeType.RAND_LAYER:
            if self.training:
                return self.random_layer_exchange(featA, featB, thresh)
            return self.layer_exchange(featA, featB, p)

        layer_ids = layers if layers is not None else list(range(len(featA)))

        if mode == ExchangeType.CHANNEL or mode == ExchangeType.RAND_CHANNEL:
            for i in layer_ids:
                if mode == ExchangeType.RAND_CHANNEL and self.training:
                    featA[i], featB[i] = self.random_channel_exchange(featA[i], featB[i], thresh)
                else:
                    featA[i], featB[i] = self.channel_exchange(featA[i], featB[i], p)
            return featA, featB

        if mode == ExchangeType.SPATIAL or mode == ExchangeType.RAND_SPATIAL:
            for i in layer_ids:
                if mode == ExchangeType.RAND_SPATIAL and self.training:
                    featA[i], featB[i] = self.random_spatial_exchange(featA[i], featB[i], thresh)
                else:
                    featA[i], featB[i] = self.spatial_exchange(featA[i], featB[i], p)
            return featA, featB

        if mode == ExchangeType.FDSE:
            raise ValueError(f"{mode.value} is trainable and must be called from CADRNet's soft-exchange module.")

        return self.layer_exchange(featA, featB)


if __name__ == '__main__':
    A = [torch.randn(1, 16, 32, 32) for _ in range(5)]
    B = [torch.randn_like(t) for t in A]
    exch = FeatureExchanger(training=True)
    A2, B2 = exch.exchange(A, B, mode=ExchangeType.CHANNEL, p=2, layers=[2, 3])
    print([t.shape for t in A2], [t.shape for t in B2])
