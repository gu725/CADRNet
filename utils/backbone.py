import timm
import os


LOCAL_PRETRAIN_FILES = {
    'resnet18': 'resnet18.a1_in1k/pytorch_model.bin',
    'resnet34': 'resnet34.a1_in1k/pytorch_model.bin',
    'resnet50': 'resnet50.a1_in1k/pytorch_model.bin',
    'mobilenetv4_conv_small': 'mobilenetv4_conv_small.e3600_r256_in1k/pytorch_model.bin',
    'efficientnet_b0': 'efficientnet_b0.ra_in1k/pytorch_model.bin',
    'efficientnet_b4': 'efficientnet_b4.ra2_in1k/pytorch_model.bin',
    'efficientnet_b5': 'efficientnet_b5.sw_in12k_ft_in1k/pytorch_model.bin',
    'hrnet_w18': 'hrnet_w18.ms_aug_in1k/pytorch_model.bin',
    'hrnet_w32': 'hrnet_w32.ms_in1k/pytorch_model.bin',
    'hrnet_w64': 'hrnet_w64.ms_in1k/pytorch_model.bin',
    'convnext_base': 'convnext_base.fb_in22k_ft_in1k/pytorch_model.bin',
    'convnext_tiny': 'convnext_tiny.in12k_ft_in1k/pytorch_model.bin',
    'mambaout_base': 'mambaout_base.in1k/pytorch_model.bin',
    'mambaout_small': 'mambaout_small.in1k/pytorch_model.bin',
    'mambaout_tiny': 'mambaout_tiny.in1k/pytorch_model.bin',
    'swinv2_base_window8_256': 'swinv2_base_window8_256.ms_in1k/pytorch_model.bin',
    'swinv2_small_window8_256': 'swinv2_small_window8_256.ms_in1k/pytorch_model.bin',
    'swinv2_tiny_window8_256': 'swinv2_tiny_window8_256.ms_in1k/pytorch_model.bin',
}

MODEL_ALIASES = {
    'mobilenetv4': 'mobilenetv4_conv_small',
}


def _create_feature_model(model_name, pretrained=False):
    if not pretrained:
        return timm.create_model(model_name, pretrained=False, features_only=True)

    pretrain_root = os.environ.get("PRETRAIN")
    local_file = LOCAL_PRETRAIN_FILES.get(model_name)

    if pretrain_root and local_file:
        weight_path = os.path.join(pretrain_root, local_file)
        if os.path.exists(weight_path):
            return timm.create_model(
                model_name,
                pretrained=True,
                pretrained_cfg_overlay={"file": weight_path},
                features_only=True,
            )
        print(f"[SEED] Local pretrained file not found: {weight_path}")

    try:
        return timm.create_model(model_name, pretrained=True, features_only=True)
    except Exception as exc:
        print(
            f"[SEED] Could not load pretrained weights for {model_name}: {exc}. "
            "Falling back to pretrained=False."
        )
        return timm.create_model(model_name, pretrained=False, features_only=True)


def _build_fpn_config(model, num_outs=4, out_channels=256):
    in_channels = model.feature_info.channels()
    if len(in_channels) < num_outs:
        raise ValueError(f"Backbone returned only {len(in_channels)} feature maps; need at least {num_outs}.")
    return {
        'type': 'FPN',
        'in_channels': in_channels[-num_outs:],
        'out_channels': out_channels,
        'num_outs': num_outs,
    }, len(in_channels)



def build_backbone(model_name='resnet50', pretrained=False):
    if model_name is None:
        model_name = 'hrnet_w64'
    model_name = MODEL_ALIASES.get(model_name, model_name)
    if model_name not in LOCAL_PRETRAIN_FILES:
        raise ValueError(f"Unsupported backbone '{model_name}'. Available: {sorted(LOCAL_PRETRAIN_FILES)}")

    model = _create_feature_model(model_name, pretrained=pretrained)
    FPN_DICT, num_stages = _build_fpn_config(model)
    dim_change = model_name.startswith('mambaout') or model_name.startswith('swinv2')
    return model, num_stages, FPN_DICT, dim_change
