import torch
import torch.nn as nn
import torch.nn.functional as F

from cadrnet.utils.backbone import build_backbone
from cadrnet.utils.decode_block import ResDecodeBlock, SwinDecodeBlock
from cadrnet.utils.exchange import (
    ExchangeType,
    FeatureExchanger,
    FrequencyDecoupledSpectralExchange,
)
from cadrnet.utils.fpn import build_fpn
from cadrnet.modules.cdam_encoder import CDAMv4
from cadrnet.modules.gdcr_decoder import GDCR, MultiScaleGDCR
from cadrnet.modules.multiscale_cdam_prior_fusion import MultiScaleCDAMPriorFusion


class CADRNet(nn.Module):
    """CADRNet change detection network."""

    def __init__(
        self,
        model_name='hrnet_w64',
        exchange_mode=ExchangeType.LAYER,
        exchange_prob=0.5,
        exchange_step=2,
        exchange_layers=None,
        decoder_scale='single',
        cdam=1,
        decoder_refiner='gdcr',
        gdcr_eta_init=0.1,
        gdcr_edge_mode='sobel',
        gdcr_return_aux=False,
        use_multiscale_cdam_prior=True,
        cdam_mode='residual',
        pretrained=False,
    ):
        super().__init__()
        if model_name is None:
            model_name = 'hrnet_w64'

        self.backbone_name = model_name
        self.model, self.num_stages, fpn_cfg, self.dim_change = build_backbone(
            model_name=model_name,
            pretrained=pretrained,
        )
        self.fpn = build_fpn(fpn_cfg)
        self.decoder_scale = str(decoder_scale).strip().lower()
        if self.decoder_scale not in {'single', 'multi'}:
            raise ValueError("decoder_scale must be either 'single' or 'multi'.")
        self.decoder_feature_count = 1 if self.decoder_scale == 'single' else 4
        self.use_cdam = bool(int(cdam))
        decoder_refiner = str(decoder_refiner).strip().lower()
        if decoder_refiner == 'auto':
            decoder_refiner = 'gdcr'
        if decoder_refiner not in {'none', 'gdcr'}:
            raise ValueError("decoder_refiner must be one of: auto, none, gdcr.")
        self.decoder_refiner = decoder_refiner
        self.use_gdcr = self.decoder_refiner == 'gdcr'
        self.use_multiscale_cdam_prior = bool(int(use_multiscale_cdam_prior))
        self.cdam_module_cls = CDAMv4
        self.cdam_mode = str(cdam_mode).lower()
        if self.cdam_mode not in {'residual', 'suppressive'}:
            raise ValueError("cdam_mode must be either 'residual' or 'suppressive'.")

        if isinstance(exchange_mode, str):
            exchange_mode = ExchangeType(exchange_mode.lower())
        self.exchange_mode = exchange_mode
        self.exchange_prob = exchange_prob
        self.exchange_step = exchange_step
        self.exchange_layers = exchange_layers
        self.exchanger = FeatureExchanger(training=self.training)
        self.fdse = None

        if self.use_cdam:
            cdam_modules = []
            for in_channels in fpn_cfg['in_channels']:
                cdam_kwargs = {
                    'in_channels': in_channels,
                    'modulation_mode': self.cdam_mode,
                    'use_aux_loss': True,
                }
                cdam_modules.append(self.cdam_module_cls(**cdam_kwargs))
            self.cdam_modules = nn.ModuleList(cdam_modules)
        else:
            self.cdam_modules = nn.ModuleList()

        if self.exchange_mode == ExchangeType.FDSE:
            self.fdse = FrequencyDecoupledSpectralExchange(fpn_cfg['in_channels'])

        out_channels = fpn_cfg['out_channels']
        decode_block = SwinDecodeBlock if 'swinv2' in model_name else ResDecodeBlock
        self.decoder_blocks = nn.ModuleList(
            [decode_block(in_channels=out_channels) for _ in range(4)]
        )
        self.decode_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.multiscale_cdam_fusion = None
        if self.use_gdcr and self.use_multiscale_cdam_prior:
            self.multiscale_cdam_fusion = MultiScaleCDAMPriorFusion(
                in_channels_list=fpn_cfg['in_channels'],
                decoder_channels=out_channels,
                hidden_channels=max(out_channels // 4, 16),
            )
            self.gdcr = GDCR(
                decoder_channels=out_channels,
                encoder_channels=out_channels,
                eta_init=gdcr_eta_init,
                edge_mode=gdcr_edge_mode,
                return_aux=bool(int(gdcr_return_aux)),
                use_aux_loss=True,
            )
        elif self.use_gdcr:
            self.gdcr = MultiScaleGDCR(
                decoder_channels_list=[out_channels] * self.decoder_feature_count,
                encoder_channels_list=fpn_cfg['in_channels'][: self.decoder_feature_count],
                eta_init=gdcr_eta_init,
                edge_mode=gdcr_edge_mode,
                return_aux=bool(int(gdcr_return_aux)),
                use_aux_loss=True,
            )
        else:
            self.gdcr = None
        self.conv_seg = nn.Conv2d(out_channels, 2, kernel_size=1)

    def _extract_features(self, x):
        features = self.model(x)[self.num_stages - 4:]
        if self.dim_change:
            features = [feature.permute(0, 3, 1, 2) for feature in features]
        return features

    def _apply_cdam(self, feat_a, feat_b, gt=None):
        if not self.use_cdam:
            return feat_a, feat_b, {}, None, None

        out_a = []
        out_b = []
        aux_items = []
        attention_maps = []
        style_stats = []
        use_aux = self.training and gt is not None

        for module, feature_a, feature_b in zip(self.cdam_modules, feat_a, feat_b):
            result = module(feature_a, feature_b, gt=gt if use_aux else None)
            if len(result) == 3:
                feature_a, feature_b, aux_dict = result
                attention_maps.append(aux_dict.get("change_attention"))
                style_stats.append(aux_dict.get("domain_stat"))
                if use_aux:
                    aux_items.append(aux_dict)
            else:
                feature_a, feature_b = result
                attention_maps.append(None)
                style_stats.append(None)
            out_a.append(feature_a)
            out_b.append(feature_b)

        aux = {}
        if aux_items:
            for key in (
                "loss_cdam",
                "loss_align",
                "loss_spatial",
                "loss_sparsity",
                "loss_domain",
                "loss_channel",
            ):
                values = [item[key] for item in aux_items if key in item]
                if values:
                    aux[key] = torch.stack(values).mean()
        if not any(attention is not None for attention in attention_maps):
            attention_maps = None
        if not any(style_stat is not None for style_stat in style_stats):
            style_stats = None
        return out_a, out_b, aux, attention_maps, style_stats

    @staticmethod
    def _compute_style_stat(f1, f2, eps=1e-6):
        f1_stats = f1.float()
        f2_stats = f2.float()
        mu1 = f1_stats.mean(dim=(2, 3), keepdim=True)
        mu2 = f2_stats.mean(dim=(2, 3), keepdim=True)
        std1 = torch.sqrt(((f1_stats - mu1) ** 2).mean(dim=(2, 3), keepdim=True) + eps)
        std2 = torch.sqrt(((f2_stats - mu2) ** 2).mean(dim=(2, 3), keepdim=True) + eps)
        return (torch.abs(mu1 - mu2) + torch.abs(std1 - std2)).to(dtype=f1.dtype)

    @staticmethod
    def _has_full_guidance(cdam_attention, style_stats):
        if cdam_attention is None or style_stats is None:
            return False
        if len(cdam_attention) < 4 or len(style_stats) < 4:
            return False
        return all(item is not None for item in cdam_attention[:4]) and all(
            item is not None for item in style_stats[:4]
        )

    @staticmethod
    def _merge_gdcr_branch_aux(aux_a, aux_b):
        merged = {}
        loss_keys = (
            "loss_gdcr",
            "loss_gdcr_align",
            "loss_gdcr_sparsity",
            "loss_gdcr_boundary",
            "loss_gdcr_gate",
        )
        for key in loss_keys:
            values = [aux[key] for aux in (aux_a, aux_b) if key in aux]
            if values:
                merged[key] = torch.stack(values).mean()

        for prefix, aux in (("a", aux_a), ("b", aux_b)):
            for key, value in aux.items():
                if key not in loss_keys:
                    merged[f"gdcr_{prefix}_{key}"] = value
        return merged

    def _apply_gdcr(
        self,
        decoded_a,
        decoded_b,
        encoder_a,
        encoder_b,
        fpn_a=None,
        fpn_b=None,
        cdam_attention=None,
        style_stats=None,
        gt=None,
    ):
        if not self.use_gdcr or self.gdcr is None:
            return decoded_a, decoded_b, {}

        if self.use_multiscale_cdam_prior:
            if self.multiscale_cdam_fusion is None:
                raise RuntimeError("Multi-scale CDAM prior fusion is not initialized.")
            if fpn_a is None or fpn_b is None:
                raise RuntimeError("GDCR multi-scale prior path requires FPN features.")

            f1_hat = fpn_a[0]
            f2_hat = fpn_b[0]
            if self._has_full_guidance(cdam_attention, style_stats):
                a_c, s_d = self.multiscale_cdam_fusion(
                    cdam_attention[:4],
                    style_stats[:4],
                    target_size=decoded_a[0].shape[-2:],
                )
            else:
                a_c = torch.abs(f1_hat - f2_hat)
                s_d = self._compute_style_stat(f1_hat, f2_hat)

            result_a = self.gdcr(decoded_a[0], f1_hat, f2_hat, a_c, s_d, gt=gt)
            result_b = self.gdcr(decoded_b[0], f1_hat, f2_hat, a_c, s_d, gt=gt)
            if isinstance(result_a, tuple):
                refined_a, aux_a = result_a
                refined_b, aux_b = result_b
                aux = self._merge_gdcr_branch_aux(aux_a, aux_b)
            else:
                refined_a = result_a
                refined_b = result_b
                aux = {}

            decoded_a = [refined_a] + list(decoded_a[1:])
            decoded_b = [refined_b] + list(decoded_b[1:])
            return decoded_a, decoded_b, aux

        result = self.gdcr(
            decoded_a,
            decoded_b,
            encoder_a,
            encoder_b,
            attention_maps=cdam_attention,
            style_stats=style_stats,
            gt=gt,
        )
        if len(result) == 3:
            refined_a, refined_b, aux = result
            return refined_a, refined_b, aux

        refined_a, refined_b = result
        return refined_a, refined_b, {}

    def _exchange_features(self, feat_a, feat_b, cdam_attention=None, style_stats=None):
        if self.exchange_mode == ExchangeType.FDSE:
            if self.fdse is None:
                raise RuntimeError("FDSE module is not initialized.")
            out_a, out_b = self.fdse(
                feat_a,
                feat_b,
                attention_maps=cdam_attention,
                style_stats=style_stats,
                layers=self.exchange_layers,
            )
            return out_a, out_b, {}

        self.exchanger.training = self.training
        out_a, out_b = self.exchanger.exchange(
            feat_a,
            feat_b,
            mode=self.exchange_mode,
            thresh=self.exchange_prob,
            p=self.exchange_step,
            layers=self.exchange_layers,
        )
        return out_a, out_b, {}

    def decode_stage(self, feature_list, return_multiscale=False, output_size=None):
        del output_size
        x1, x2, x3, x4 = feature_list

        # Shared decoder: refine deepest feature, then upsample and fuse skip features with pixel-wise add.
        d4 = self.decoder_blocks[0](x4)
        x = F.interpolate(d4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = x + x3

        d3 = self.decoder_blocks[1](x)
        x = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = x + x2

        d2 = self.decoder_blocks[2](x)
        x = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = x + x1

        d1 = self.decoder_blocks[3](x)
        if return_multiscale:
            if self.decoder_scale == 'single':
                return [d1]
            return [d1, d2, d3, d4]
        return d1

    def decode_logits(self, x, output_size):
        x = self.decode_head(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return self.conv_seg(x)

    def merge_logits(self, logits_list):
        return torch.stack(list(logits_list), dim=0).mean(dim=0)

    def forward(self, xA, xB, gt=None):
        aux = {}
        xA_list = self._extract_features(xA)
        xB_list = self._extract_features(xB)

        xA_list, xB_list, cdam_aux, cdam_attention, style_stats = self._apply_cdam(xA_list, xB_list, gt=gt)
        cdam_feat_a = list(xA_list)
        cdam_feat_b = list(xB_list)
        aux.update(cdam_aux)

        xA_list, xB_list, exchange_aux = self._exchange_features(
            xA_list,
            xB_list,
            cdam_attention=cdam_attention,
            style_stats=style_stats,
        )
        aux.update(exchange_aux)
        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        if self.use_gdcr:
            decoded_a = self.decode_stage(xA_list, return_multiscale=True, output_size=xA.shape[2:])
            decoded_b = self.decode_stage(xB_list, return_multiscale=True, output_size=xB.shape[2:])
            decoded_a, decoded_b, gdcr_aux = self._apply_gdcr(
                decoded_a,
                decoded_b,
                cdam_feat_a,
                cdam_feat_b,
                fpn_a=xA_list,
                fpn_b=xB_list,
                cdam_attention=cdam_attention,
                style_stats=style_stats,
                gt=gt,
            )
            aux.update(gdcr_aux)
            outA = decoded_a[0]
            outB = decoded_b[0]
        else:
            outA = self.decode_stage(xA_list, output_size=xA.shape[2:])
            outB = self.decode_stage(xB_list, output_size=xB.shape[2:])
        outA = self.decode_logits(outA, output_size=xA.shape[2:])
        outB = self.decode_logits(outB, output_size=xB.shape[2:])
        if aux:
            return {"logits": [outA, outB], "aux": aux}
        return [outA, outB]


if __name__ == '__main__':
    model = CADRNet(model_name="resnet18")
    xA = torch.randn(2, 3, 256, 256)
    xB = torch.randn(2, 3, 256, 256)
    output = model(xA, xB)
    outA, outB = output["logits"] if isinstance(output, dict) else output
    print(outA.shape, outB.shape)
    print("Model forward pass successful.")
