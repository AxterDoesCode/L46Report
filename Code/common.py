# common.py
# Shared Brevitas quantisation injector configurations for the CNV model.
# CommonQuant defines the fixed-point baseline; CommonWeightQuant and
# CommonActQuant specialise it for weights and activations respectively.

from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import WeightQuantSolver


# Base quantisation config: constant bit-width, symmetric fixed-point,
# round-to-nearest, per-tensor scaling, narrow range.
class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    # Select FP pass-through, 1-bit binary, or integer quantisation.
    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


# Weight quantisation: fixed scale of 1.0 maps weights to [-1, 1].
class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


# Activation quantisation: clips to [-1.0, 1.0].
class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0
