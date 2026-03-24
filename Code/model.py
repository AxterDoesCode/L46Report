# model.py
# Defines the CNV quantised CNN architecture for CIFAR-10 classification
# using Brevitas quantisation-aware training layers.

import torch
import torch.nn as nn

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity

from .common import CommonActQuant, CommonWeightQuant
from .tensor_norm import TensorNorm


# Architecture configuration
# (output_channels, apply_maxpool)
CONV_CHANNELS = [
    (64, False),
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False),
]

# (in_features, out_features) for the hidden FC layers
FC_LAYERS = [
    (256, 512),
    (512, 512),
]

FINAL_FC_INPUT = 512
KERNEL_SIZE = 3
POOL_SIZE = 2


class CNV(nn.Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # Input quantisation: fixed-point Q1.7 format covering [-1, 1 - 2^-7]
        self.conv_layers.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO
            )
        )

        # Convolutional feature extractor
        # Each block: QuantConv2d -> BatchNorm2d -> QuantIdentity [-> MaxPool2d]
        for out_ch, use_pool in CONV_CHANNELS:

            self.conv_layers.append(
                QuantConv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=KERNEL_SIZE,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            in_ch = out_ch

            self.conv_layers.append(nn.BatchNorm2d(in_ch, eps=1e-4))
            self.conv_layers.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width
                )
            )

            if use_pool:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

        # Fully connected classifier
        # Each block: QuantLinear -> BatchNorm1d -> QuantIdentity
        for in_features, out_features in FC_LAYERS:

            self.fc_layers.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )
            self.fc_layers.append(nn.BatchNorm1d(out_features, eps=1e-4))
            self.fc_layers.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width
                )
            )

        # Final classification layer + TensorNorm to stabilise logits
        self.fc_layers.append(
            QuantLinear(
                in_features=FINAL_FC_INPUT,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width
            )
        )
        self.fc_layers.append(TensorNorm())

        # Initialise all quantised weights uniformly in [-1, 1]
        for module in self.modules():
            if isinstance(module, (QuantConv2d, QuantLinear)):
                torch.nn.init.uniform_(module.weight.data, -1, 1)

    # Clamp weights to [min_val, max_val] after each optimiser step.
    def clip_weights(self, min_val, max_val):
        for layer in self.conv_layers:
            if isinstance(layer, QuantConv2d):
                layer.weight.data.clamp_(min_val, max_val)

        for layer in self.fc_layers:
            if isinstance(layer, QuantLinear):
                layer.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        # Rescale input from [0, 1] to [-1, 1] to match the quantisation range
        x = 2.0 * x - 1.0

        for layer in self.conv_layers:
            x = layer(x)

        # Flatten spatial feature maps into a 1-D vector per sample
        x = x.view(x.size(0), -1)

        for layer in self.fc_layers:
            x = layer(x)

        return x


# Builds the CNV model from a ConfigParser object.
def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')

    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')

    return CNV(
        num_classes=num_classes,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        in_ch=in_channels
    )
