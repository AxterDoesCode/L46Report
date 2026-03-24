# File: model.py
import torch
import torch.nn as nn

# Brevitas quantization layers
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d, QuantLinear, QuantIdentity

# Local project utilities
from .common import CommonActQuant, CommonWeightQuant
from .tensor_norm import TensorNorm


# -----------------------------
# Architecture configuration
# -----------------------------

# (output_channels, apply_maxpool)
CONV_CHANNELS = [
    (64, False),
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False)
]

# Fully connected layer sizes
FC_LAYERS = [
    (256, 512),
    (512, 512)
]

FINAL_FC_INPUT = 512
KERNEL_SIZE = 3
POOL_SIZE = 2


# -----------------------------
# Main CNN Model
# -----------------------------
class CNV(nn.Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super().__init__()

        # These will store sequential layers
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # ----------------------------------
        # Input Quantization Layer
        # ----------------------------------
        # This simulates quantizing the input image (e.g., fixed-point Q1.7 format)
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

        # ----------------------------------
        # Convolutional Feature Extractor
        # ----------------------------------
        for out_ch, use_pool in CONV_CHANNELS:

            # Quantized convolution (weights are quantized)
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

            # Update input channels for next layer
            in_ch = out_ch

            # BatchNorm stabilizes training
            self.conv_layers.append(nn.BatchNorm2d(in_ch, eps=1e-4))

            # Quantized activation (like ReLU but quantized)
            self.conv_layers.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width
                )
            )

            # Optional downsampling
            if use_pool:
                self.conv_layers.append(
                    nn.MaxPool2d(kernel_size=POOL_SIZE)
                )

        # ----------------------------------
        # Fully Connected Layers
        # ----------------------------------
        for in_features, out_features in FC_LAYERS:

            # Quantized linear layer
            self.fc_layers.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width
                )
            )

            # BatchNorm for stability
            self.fc_layers.append(nn.BatchNorm1d(out_features, eps=1e-4))

            # Quantized activation
            self.fc_layers.append(
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bit_width
                )
            )

        # ----------------------------------
        # Final Classification Layer
        # ----------------------------------
        self.fc_layers.append(
            QuantLinear(
                in_features=FINAL_FC_INPUT,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width
            )
        )

        # Normalization of final outputs (helps training stability)
        self.fc_layers.append(TensorNorm())

        # ----------------------------------
        # Weight Initialization
        # ----------------------------------
        # Initialize all quantized weights uniformly in [-1, 1]
        for module in self.modules():
            if isinstance(module, (QuantConv2d, QuantLinear)):
                torch.nn.init.uniform_(module.weight.data, -1, 1)

    # ----------------------------------
    # Optional: Weight Clipping
    # ----------------------------------
    def clip_weights(self, min_val, max_val):
        """
        Clamp weights to a fixed range.
        Useful for quantization stability.
        """
        for layer in self.conv_layers:
            if isinstance(layer, QuantConv2d):
                layer.weight.data.clamp_(min_val, max_val)

        for layer in self.fc_layers:
            if isinstance(layer, QuantLinear):
                layer.weight.data.clamp_(min_val, max_val)

    # ----------------------------------
    # Forward Pass
    # ----------------------------------
    def forward(self, x):
        """
        Defines how data flows through the network.
        """

        # Rescale input from [0, 1] → [-1, 1]
        x = 2.0 * x - 1.0

        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten feature maps into a vector
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        return x


# -----------------------------
# Model Factory Function
# -----------------------------
def cnv(cfg):
    """
    Builds the model from a config object.
    """

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
