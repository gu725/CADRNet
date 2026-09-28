from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


GDCROutput = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
MultiScaleGDCROutput = Union[
    Tuple[List[torch.Tensor], List[torch.Tensor]],
    Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]],
]


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU block for spatial feature tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GateBlock(nn.Module):
    """Generate a sigmoid gate with Conv3x3-BN-ReLU-Conv1x1."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden_channels = max(out_channels // 2, 16)
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.block(x))


class SobelEdgeExtractor(nn.Module):
    """Fixed Sobel edge extractor for a single-channel spatial prior."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError("SobelEdgeExtractor expects a [B,1,H,W] tensor.")
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
        edge_x = torch.abs(F.conv2d(x, sobel_x, padding=1))
        edge_y = torch.abs(F.conv2d(x, sobel_y, padding=1))
        return edge_x + edge_y


class GDCR(nn.Module):
    """Guided Dual-branch Change Refinement Module.

    GDCR refines one decoder feature map before the final prediction head. It
    consumes the current decoder feature and CDAM-side guidance, then rebuilds
    change features through complementary region and boundary branches.

    Args:
        decoder_channels: C_D, channels of x_d and the returned feature.
        encoder_channels: C_E, channels of f1_hat, f2_hat, a_c, and s_d.
        dilation_rates: dilation rates for the region reconstruction branch.
        eta_init: initial value of the learnable residual scale eta.
        edge_mode: "sobel" for fixed Sobel prior, or "learnable" for a learned
            3x3 single-channel edge branch.
        return_aux: when True, return intermediate tensors for visualization
            and debugging.

    Inputs:
        x_d: [B, C_D, H, W], decoder feature.
        f1_hat: [B, C_E, H_e, W_e], CDAM-enhanced temporal feature 1.
        f2_hat: [B, C_E, H_e, W_e], CDAM-enhanced temporal feature 2.
        a_c: [B, C_E, H_e, W_e], CDAM change-attention prior.
        s_d: [B, C_E, 1, 1], CDAM domain/style statistic difference.

    Output:
        x_hat: [B, C_D, H, W], refined decoder feature. If return_aux=True,
        returns (x_hat, aux_dict).
    """

    def __init__(
        self,
        decoder_channels: int,
        encoder_channels: int,
        dilation_rates: Sequence[int] = (1, 3, 5),
        eta_init: float = 0.1,
        edge_mode: str = "sobel",
        return_aux: bool = False,
        use_aux_loss: bool = False,
        lambda_sparsity: float = 0.01,
        lambda_boundary: float = 0.1,
        lambda_gate: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if encoder_channels <= 0:
            raise ValueError("encoder_channels must be positive.")
        if not dilation_rates:
            raise ValueError("dilation_rates must contain at least one value.")

        edge_mode = str(edge_mode).lower()
        if edge_mode not in {"sobel", "learnable"}:
            raise ValueError("edge_mode must be either 'sobel' or 'learnable'.")

        self.decoder_channels = int(decoder_channels)
        self.encoder_channels = int(encoder_channels)
        self.dilation_rates = tuple(int(rate) for rate in dilation_rates)
        self.edge_mode = edge_mode
        self.return_aux = bool(return_aux)
        self.use_aux_loss = bool(use_aux_loss)
        self.lambda_sparsity = float(lambda_sparsity)
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_gate = float(lambda_gate)
        self.eps = float(eps)
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))

        c_d = self.decoder_channels
        c_e = self.encoder_channels
        branch_hidden = max(c_d // 4, 16)

        self.phi_delta = ConvBNReLU(c_e, c_d, kernel_size=3)
        self.phi0 = ConvBNReLU(c_d * 2, c_d, kernel_size=3)
        self.phi_c = nn.Conv2d(c_e, c_d, kernel_size=1, bias=True)
        self.phi_s = nn.Sequential(
            nn.Conv2d(c_e, c_d, kernel_size=1, bias=False),
            nn.GroupNorm(1, c_d),
            nn.ReLU(inplace=False),
        )

        self.region_branches = nn.ModuleList(
            ConvBNReLU(c_d, c_d, kernel_size=3, dilation=rate)
            for rate in self.dilation_rates
        )
        self.phi_r = ConvBNReLU(c_d * len(self.dilation_rates), c_d, kernel_size=1, padding=0)
        self.phi_gr = GateBlock(c_d * 4, c_d)
        self.phi_rr = ConvBNReLU(c_d, c_d, kernel_size=3)

        if edge_mode == "sobel":
            self.edge_extractor = SobelEdgeExtractor()
        else:
            self.edge_extractor = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)

        self.phi_b = ConvBNReLU(c_d * 2 + 1, c_d, kernel_size=3)
        self.phi_gb = GateBlock(c_d + 2, c_d)
        self.phi_bb = ConvBNReLU(c_d, c_d, kernel_size=3)

        self.phi_b2r = nn.Conv2d(c_d, c_d, kernel_size=3, padding=2, dilation=2, bias=True)
        self.phi_br = ConvBNReLU(c_d, c_d, kernel_size=3)
        self.phi_r2b = nn.Conv2d(c_d, c_d, kernel_size=3, padding=1, bias=True)
        self.phi_rb = ConvBNReLU(c_d, c_d, kernel_size=3)

        self.phi_f = nn.Sequential(
            nn.Conv2d(c_d * 3, branch_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(branch_hidden, c_d * 2, kernel_size=1, bias=True),
        )
        self.phi_o = GateBlock(c_d * 4, c_d)
        self.phi_out = nn.Sequential(
            ConvBNReLU(c_d, c_d, kernel_size=3),
            nn.Conv2d(c_d, c_d, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _prepare_gt(
        self,
        gt: Optional[torch.Tensor],
        size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if gt is None:
            return None
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        elif gt.dim() == 4:
            if gt.shape[1] > 1:
                gt = gt.argmax(dim=1, keepdim=True)
        else:
            raise ValueError("gt must have shape [B,H,W], [B,1,H,W], or one-hot [B,C,H,W].")

        gt = gt.to(device=device, dtype=torch.float32)
        if gt.shape[-2:] != size:
            gt = F.interpolate(gt, size=size, mode="nearest")
        gt = (gt > 0.5).float() if gt.max() <= 1.0 else (gt > 0.0).float()
        return gt.to(dtype=dtype)

    def _make_boundary_target(self, gt: torch.Tensor) -> torch.Tensor:
        if gt.dim() != 4 or gt.shape[1] != 1:
            raise ValueError("gt must have shape [B,1,H,W] before boundary extraction.")
        dilated = F.max_pool2d(gt, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-gt, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _bce_from_prob(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(prob.float().clamp(self.eps, 1.0 - self.eps))
        return F.binary_cross_entropy_with_logits(logits, target.float())

    def _normalize_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(0.0)
        denom = value.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        return (value / denom).clamp(0.0, 1.0)

    def _compute_aux_loss(
        self,
        gt: Optional[torch.Tensor],
        change_prior: torch.Tensor,
        boundary_prior: torch.Tensor,
        final_gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if gt is None:
            zero = change_prior.sum() * 0.0
            return {
                "loss_gdcr": zero,
                "loss_gdcr_align": zero,
                "loss_gdcr_sparsity": zero,
                "loss_gdcr_boundary": zero,
                "loss_gdcr_gate": zero,
            }

        target = gt.float()
        gate_prob = final_gate.float().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        boundary_prob = self._normalize_prob(boundary_prior)
        boundary_target = self._make_boundary_target(target)

        loss_align = self._bce_from_prob(change_prior, target)
        loss_sparsity = change_prior.float().mean()
        loss_boundary = self._bce_from_prob(boundary_prob, boundary_target)
        loss_gate = self._bce_from_prob(gate_prob, target)
        loss_gdcr = (
            loss_align
            + self.lambda_sparsity * loss_sparsity
            + self.lambda_boundary * loss_boundary
            + self.lambda_gate * loss_gate
        )
        return {
            "loss_gdcr": loss_gdcr,
            "loss_gdcr_align": loss_align,
            "loss_gdcr_sparsity": loss_sparsity,
            "loss_gdcr_boundary": loss_boundary,
            "loss_gdcr_gate": loss_gate,
        }

    def _check_and_align(
        self,
        x_d: torch.Tensor,
        f1_hat: torch.Tensor,
        f2_hat: torch.Tensor,
        a_c: torch.Tensor,
        s_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_d.dim() != 4:
            raise ValueError("x_d must have shape [B,C_D,H,W].")
        if x_d.shape[1] != self.decoder_channels:
            raise ValueError(
                f"x_d expected {self.decoder_channels} channels, got {x_d.shape[1]}."
            )
        for name, tensor in (("f1_hat", f1_hat), ("f2_hat", f2_hat), ("a_c", a_c)):
            if tensor.dim() != 4:
                raise ValueError(f"{name} must have shape [B,C_E,H,W].")
            if tensor.shape[0] != x_d.shape[0]:
                raise ValueError(f"{name} batch size must match x_d.")
            if tensor.shape[1] != self.encoder_channels:
                raise ValueError(
                    f"{name} expected {self.encoder_channels} channels, got {tensor.shape[1]}."
                )
        if f1_hat.shape != f2_hat.shape:
            raise ValueError(f"f1_hat and f2_hat must have identical shapes, got {f1_hat.shape} and {f2_hat.shape}.")
        if a_c.shape[0] != x_d.shape[0] or a_c.shape[1] != self.encoder_channels:
            raise ValueError("a_c batch/channel dimensions must match x_d and encoder_channels.")
        if s_d.dim() != 4 or s_d.shape[0] != x_d.shape[0] or s_d.shape[1] != self.encoder_channels:
            raise ValueError("s_d must have shape [B,C_E,1,1].")
        if s_d.shape[-2:] != (1, 1):
            raise ValueError(f"s_d spatial shape must be [1,1], got {s_d.shape[-2:]}.")

        target_size = x_d.shape[-2:]

        def _align(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.to(device=x_d.device, dtype=x_d.dtype)
            if tensor.shape[-2:] != target_size:
                tensor = F.interpolate(
                    tensor,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            return tensor

        return x_d, _align(f1_hat), _align(f2_hat), _align(a_c), s_d.to(device=x_d.device, dtype=x_d.dtype)

    def forward(
        self,
        x_d: torch.Tensor,
        f1_hat: torch.Tensor,
        f2_hat: torch.Tensor,
        a_c: torch.Tensor,
        s_d: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
    ) -> GDCROutput:
        x_d, f1_hat, f2_hat, a_c, s_d = self._check_and_align(x_d, f1_hat, f2_hat, a_c, s_d)
        h, w = x_d.shape[-2:]

        d_e = torch.abs(f1_hat - f2_hat)
        e_delta = self.phi_delta(d_e)
        x0 = self.phi0(torch.cat([x_d, e_delta], dim=1))

        p_c = torch.sigmoid(self.phi_c(a_c))
        p_s = p_c.mean(dim=1, keepdim=True)

        u_s_vector = torch.sigmoid(self.phi_s(s_d))
        u_s = u_s_vector.expand(-1, -1, h, w)

        region_feats = [branch(x0) for branch in self.region_branches]
        r_m = self.phi_r(torch.cat(region_feats, dim=1))
        g_r = self.phi_gr(torch.cat([r_m, e_delta, p_c, u_s], dim=1))
        r_f = self.phi_rr(r_m * g_r)

        if self.edge_mode == "sobel":
            b_p = self.edge_extractor(p_s)
        else:
            b_p = torch.abs(self.edge_extractor(p_s))

        h_e = e_delta - F.avg_pool2d(e_delta, kernel_size=3, stride=1, padding=1)
        h_d = x0 - F.avg_pool2d(x0, kernel_size=3, stride=1, padding=1)
        b_m = self.phi_b(torch.cat([h_e, h_d, b_p], dim=1))
        g_b = self.phi_gb(torch.cat([b_m, b_p, p_s], dim=1))
        b_f = self.phi_bb(b_m * g_b)

        g_b2r = torch.sigmoid(self.phi_b2r(b_f))
        r_star = r_f + self.phi_br(r_f * g_b2r)
        g_r2b = torch.sigmoid(self.phi_r2b(r_f))
        b_star = b_f + self.phi_rb(b_f * g_r2b)

        z_r = F.adaptive_avg_pool2d(r_star, 1)
        z_b = F.adaptive_avg_pool2d(b_star, 1)
        z_f = torch.cat([z_r, z_b, u_s_vector], dim=1)
        branch_logits = self.phi_f(z_f).view(x_d.shape[0], 2, self.decoder_channels, 1, 1)
        branch_weight = torch.softmax(branch_logits, dim=1)
        alpha_r = branch_weight[:, 0]
        alpha_b = branch_weight[:, 1]
        z = alpha_r * r_star + alpha_b * b_star

        g_o = self.phi_o(torch.cat([z, x0, p_c, u_s], dim=1))
        z_fine = z * g_o
        delta_x = self.phi_out(z_fine)
        x_hat = x0 + self.eta.to(dtype=x_d.dtype) * delta_x

        if not self.return_aux and not self.use_aux_loss:
            return x_hat

        aux: Dict[str, torch.Tensor] = {}
        if self.use_aux_loss:
            gt_resized = self._prepare_gt(
                gt=gt,
                size=(h, w),
                device=x_d.device,
                dtype=x_d.dtype,
            )
            aux.update(
                self._compute_aux_loss(
                    gt=gt_resized,
                    change_prior=p_s,
                    boundary_prior=b_p,
                    final_gate=g_o,
                )
            )

        if self.return_aux:
            aux.update({
                "x0": x0,
                "e_delta": e_delta,
                "p_c": p_c,
                "p_s": p_s,
                "style_condition": u_s_vector,
                "region_feature": r_star,
                "boundary_feature": b_star,
                "boundary_prior": b_p,
                "alpha_r": alpha_r,
                "alpha_b": alpha_b,
                "final_gate": g_o,
                "delta_x": delta_x,
            })

        return x_hat, aux


class MultiScaleGDCR(nn.Module):
    """Apply one shared GDCR block per decoder scale to two temporal branches."""

    def __init__(
        self,
        decoder_channels_list: Sequence[int],
        encoder_channels_list: Sequence[int],
        dilation_rates: Sequence[int] = (1, 3, 5),
        eta_init: float = 0.1,
        edge_mode: str = "sobel",
        return_aux: bool = False,
        use_aux_loss: bool = False,
        lambda_sparsity: float = 0.01,
        lambda_boundary: float = 0.1,
        lambda_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if len(decoder_channels_list) == 0:
            raise ValueError("decoder_channels_list must be non-empty.")
        if len(decoder_channels_list) != len(encoder_channels_list):
            raise ValueError("decoder_channels_list and encoder_channels_list must have equal length.")

        self.num_scales = len(decoder_channels_list)
        self.return_aux = bool(return_aux)
        self.use_aux_loss = bool(use_aux_loss)
        self.blocks = nn.ModuleList(
            GDCR(
                decoder_channels=int(decoder_channels),
                encoder_channels=int(encoder_channels),
                dilation_rates=dilation_rates,
                eta_init=eta_init,
                edge_mode=edge_mode,
                return_aux=return_aux,
                use_aux_loss=use_aux_loss,
                lambda_sparsity=lambda_sparsity,
                lambda_boundary=lambda_boundary,
                lambda_gate=lambda_gate,
            )
            for decoder_channels, encoder_channels in zip(decoder_channels_list, encoder_channels_list)
        )

    def _merge_loss_aux(self, aux_items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        aux: Dict[str, torch.Tensor] = {}
        for key in (
            "loss_gdcr",
            "loss_gdcr_align",
            "loss_gdcr_sparsity",
            "loss_gdcr_boundary",
            "loss_gdcr_gate",
        ):
            values = [item[key] for item in aux_items if key in item]
            if values:
                aux[key] = torch.stack(values).mean()
        return aux

    @staticmethod
    def _compute_style_stat(f1: torch.Tensor, f2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        f1_stats = f1.float()
        f2_stats = f2.float()
        mu1 = f1_stats.mean(dim=(2, 3), keepdim=True)
        mu2 = f2_stats.mean(dim=(2, 3), keepdim=True)
        std1 = torch.sqrt(((f1_stats - mu1) ** 2).mean(dim=(2, 3), keepdim=True) + eps)
        std2 = torch.sqrt(((f2_stats - mu2) ** 2).mean(dim=(2, 3), keepdim=True) + eps)
        return (torch.abs(mu1 - mu2) + torch.abs(std1 - std2)).to(dtype=f1.dtype)

    @staticmethod
    def _select_guidance(
        values: Optional[Sequence[Optional[torch.Tensor]]],
        index: int,
    ) -> Optional[torch.Tensor]:
        if values is None or index >= len(values):
            return None
        return values[index]

    def forward(
        self,
        feats_a: Sequence[torch.Tensor],
        feats_b: Sequence[torch.Tensor],
        enc_a: Sequence[torch.Tensor],
        enc_b: Sequence[torch.Tensor],
        attention_maps: Optional[Sequence[Optional[torch.Tensor]]] = None,
        style_stats: Optional[Sequence[Optional[torch.Tensor]]] = None,
        gt: Optional[torch.Tensor] = None,
    ) -> MultiScaleGDCROutput:
        if len(feats_a) != self.num_scales or len(feats_b) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} decoder feature pairs.")
        if len(enc_a) < self.num_scales or len(enc_b) < self.num_scales:
            raise ValueError(f"Expected at least {self.num_scales} encoder feature pairs.")

        out_a: List[torch.Tensor] = []
        out_b: List[torch.Tensor] = []
        aux: Dict[str, torch.Tensor] = {}
        loss_aux_items: List[Dict[str, torch.Tensor]] = []

        for idx, block in enumerate(self.blocks):
            f1_hat = enc_a[idx]
            f2_hat = enc_b[idx]
            a_c = self._select_guidance(attention_maps, idx)
            if a_c is None:
                a_c = torch.abs(f1_hat - f2_hat)
            s_d = self._select_guidance(style_stats, idx)
            if s_d is None:
                s_d = self._compute_style_stat(f1_hat, f2_hat)

            result_a = block(feats_a[idx], f1_hat, f2_hat, a_c, s_d, gt=gt)
            result_b = block(feats_b[idx], f1_hat, f2_hat, a_c, s_d, gt=gt)
            if self.return_aux or self.use_aux_loss:
                refined_a, aux_a = result_a  # type: ignore[misc]
                refined_b, aux_b = result_b  # type: ignore[misc]
                loss_aux_items.extend([aux_a, aux_b])
            else:
                refined_a = result_a  # type: ignore[assignment]
                refined_b = result_b  # type: ignore[assignment]

            if self.return_aux:
                for key, value in aux_a.items():
                    aux[f"gdcr_level{idx}_a_{key}"] = value
                for key, value in aux_b.items():
                    aux[f"gdcr_level{idx}_b_{key}"] = value
            out_a.append(refined_a)
            out_b.append(refined_b)

        if self.use_aux_loss:
            aux.update(self._merge_loss_aux(loss_aux_items))
        if self.return_aux or self.use_aux_loss:
            return out_a, out_b, aux
        return out_a, out_b


if __name__ == "__main__":
    module = GDCR(decoder_channels=128, encoder_channels=64, return_aux=True)
    x = torch.randn(2, 128, 64, 64)
    f1 = torch.randn(2, 64, 32, 32)
    f2 = torch.randn(2, 64, 32, 32)
    a = torch.randn(2, 64, 32, 32)
    s = torch.randn(2, 64, 1, 1)
    y, aux = module(x, f1, f2, a, s)
    print("out:", y.shape)
    print("aux keys:", sorted(aux))
