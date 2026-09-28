import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """A lightweight FPN replacement for the original mmseg neck."""

    def __init__(self, in_channels, out_channels, num_outs=4):
        super().__init__()
        if len(in_channels) == 0:
            raise ValueError("in_channels must contain at least one stage.")
        if num_outs < len(in_channels):
            raise ValueError("num_outs must be >= len(in_channels).")

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(in_channel, out_channels, kernel_size=1)
            for in_channel in self.in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in self.in_channels
        )

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} feature maps, but got {len(inputs)}."
            )

        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, inputs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="nearest",
            )

        outputs = [
            output_conv(lateral)
            for output_conv, lateral in zip(self.output_convs, laterals)
        ]

        while len(outputs) < self.num_outs:
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))

        return outputs[: self.num_outs]


def build_fpn(config):
    if config.get("type") != "FPN":
        raise ValueError(f"Unsupported neck type: {config.get('type')}")
    return FeaturePyramidNetwork(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        num_outs=config.get("num_outs", len(config["in_channels"])),
    )
