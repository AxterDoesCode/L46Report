# train.py
# Entry point for training and evaluating the CNV quantised network on CIFAR-10.
# Parses CLI arguments, resolves paths, constructs a Trainer, and dispatches
# to training or evaluation mode.

import argparse
import os
import sys

import torch

from .trainer import Trainer

torch.set_printoptions(precision=10)


# Add a mutually exclusive --<name> / --no_<name> flag pair to the parser.
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Return None if the string value is "None", otherwise return the string.
def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def parse_args(args):
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training using Brevitas")

    # I/O
    parser.add_argument("--datadir", default="./data/", help="Dataset location")
    parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
    parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
    parser.add_argument("--log_freq", type=int, default=10)

    # Execution modes
    parser.add_argument(
        "--evaluate", dest="evaluate", action="store_true",
        help="Evaluate model on validation set")
    parser.add_argument(
        "--resume", dest="resume", type=none_or_str,
        help="Resume from checkpoint. Overrides --pretrained flag.")
    add_bool_arg(parser, "detect_nan", default=False)

    # Compute resources
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--gpus", type=none_or_str, default=None, help="Comma separated GPUs")

    # Optimiser hyperparameters
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--lr", default=0.02, type=float, help="Learning rate")
    parser.add_argument("--optim", type=none_or_str, default="ADAM", help="Optimizer to use")
    parser.add_argument("--loss", type=none_or_str, default="SqrHinge", help="Loss function to use")
    parser.add_argument("--scheduler", default="FIXED", type=none_or_str, help="LR Scheduler")
    parser.add_argument(
        "--milestones", type=none_or_str, default='100,150,200,250', help="Scheduler milestones")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--random_seed", default=1, type=int, help="Random seed")

    # Network architecture
    parser.add_argument("--network", default="CNV_1W1A", type=str, help="Neural network")
    parser.add_argument("--pretrained", action='store_true', help="Load pretrained model")
    parser.add_argument("--strict", action='store_true', help="Strict state dictionary loading")
    parser.add_argument(
        "--state_dict_to_pth", action='store_true',
        help="Save model state_dict to a .pth file then exit")
    parser.add_argument("--export_qonnx", action='store_true', help="Export QONNX model")
    parser.add_argument("--export_qcdq_onnx", action='store_true', help="Export QCDQ ONNX model")

    return parser.parse_args(args)


# Dict subclass that also allows attribute-style key access.
class objdict(dict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def launch(cmd_args):
    args = parse_args(cmd_args)

    # Resolve relative paths against the current working directory
    path_args = ["datadir", "experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
            setattr(args, path_arg, abs_path)

    args = objdict(args.__dict__)

    # Evaluation mode never writes new output directories or checkpoints
    if args.evaluate:
        args.dry_run = True

    trainer = Trainer(args)

    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model()
            if args.export_qonnx:
                trainer.export_qonnx()
            if args.export_qcdq_onnx:
                trainer.export_qcdq_onnx()
    else:
        trainer.train_model()


def main():
    launch(sys.argv[1:])


if __name__ == "__main__":
    main()
