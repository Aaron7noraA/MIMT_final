"""Pytorch implement compression library"""
__version__ = '0.9.8'
from . import util
from .hub import (__CODER_TYPES__, AugmentedNormalizedHyperPriorCoder,
                  Coarse2FineHyperPriorCoder, CSTKContextCoder,
                  GoogleContextCoder, GoogleFactorizedCoder,
                  GoogleHyperPriorCoder, get_coder)
from .models import (CompressesModel, ContextCoder, FactorizedCoder,
                     HyperPriorCoder)
from .modules import *
from .modules import __CONV_TYPES__, __DECONV_TYPES__
from .modules.entropy_models import __CONDITIONS__


def add_coder_args(parser):
    """add_coder_args to argparser"""

    parser.add_argument(
        "--architecture", "-ARCHI", type=str, required=True, choices=__CODER_TYPES__,
        help="Coder architecture.")
    parser.add_argument(
        "--use_context", "-UC", action="store_true",
        help="Use ContextModel.")
    parser.add_argument(
        "--condition", "-C", type=str, default="Gaussian", choices=__CONDITIONS__,
        help="Condition bottelneck.")
    parser.add_argument(
        "--num_features", "-NF", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--num_filters", "-NFL", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--num_hyperpriors", "-NHP", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--Mean", "-M", action="store_true",
        help="Enable hyper-decoder to output predicted mean or not.")
    parser.add_argument(
        "--disable_signalconv", "-DS", action="store_true",
        help="Enable SignalConv or not.")
    parser.add_argument(
        "--deconv_type", "-DT", type=str, default="Signal", choices=__DECONV_TYPES__.keys(),
        help="Condition bottelneck.")
    parser.add_argument(
        "--gdn_mode", "-GM", type=str, default="standard", choices=["standard", "simplify", 'layernorm', 'pass'],
        help="Simplify GDN or not.")
    parser.add_argument(
        "--quant_mode", "-QM", type=str, default='noise', choices=EntropyModel.quant_modes,
        help="quantize with noise or round when trianing.")
    parser.add_argument(
        "--output_nought", "-ON", type=bool, default=True, #action="store_false",
        help="For ANFIC ; should be False when applying ANFIC for residual coding")
    parser.add_argument(
        "--cond_coupling", "-COND_COUP", type=bool, default=False, #action="store_false",
        help="For ANFIC ; should be True when applying ANFIC for residual coding")
    parser.add_argument(
        "--use_DQ", "-DQ", type=bool, default=True,
        help="For ANFIC ")

    return parser

def get_coder_from_args(args):
    """get_coder_from_args"""
    conv_type = "Standard" if args.disable_signalconv else SignalConv2d
    deconv_type = "Transpose" if args.disable_signalconv and args.deconv_type != "Signal" else args.deconv_type
    set_default_conv(conv_type=conv_type, deconv_type=deconv_type)
    coder = get_coder(args.architecture, num_filters=args.num_filters, num_features=args.num_features, num_hyperpriors=args.num_hyperpriors, 
                      simplify_gdn=args.gdn_mode == "simplify",
                      use_mean=args.Mean, condition=args.condition, quant_mode=args.quant_mode,
                      output_nought=args.output_nought, cond_coupling=args.cond_coupling,
                      use_DQ=args.use_DQ)
    return coder
