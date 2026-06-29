import argparse
import copy
import logging
import os
import random

import numpy as np
import torch
import yaml

from model.ADINet import ADINet

try:
    from thop import profile  # type: ignore[import-not-found]
except ImportError:
    profile = None


class Args:
    def __str__(self):
        return str(self.__dict__)


def load_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="yaml/config/default.yaml",
        help="Path to the experiment YAML file.",
    )
    cli_args = parser.parse_args()

    with open(cli_args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    args = Args()
    for category, params in config_dict.items():
        if isinstance(params, dict):
            for key, value in params.items():
                setattr(args, key, value)
        else:
            setattr(args, category, params)

    setattr(args, "config", cli_args.config)
    _set_defaults(args)
    return args


def _set_defaults(args):
    defaults = {
        "model": "ADINet",
        "window_size": None,
        "test_size": 0.2,
        "seed": 42,
        "num_workers": 0,
        "n_runs": 5,
        "checkpoint_mode": "both",
        "checkpoint_path": None,
        "model_save_format": "state_dict",
        "save_interval": 50,
        "log_interval": 1,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.model != "ADINet":
        raise ValueError("This public release only supports model='ADINet'.")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_window_size(args, sample_rate):
    if args.window_size is not None:
        return int(args.window_size)
    if int(sample_rate) == 10000:
        return 1000
    if int(sample_rate) == 6400:
        return 1024
    raise ValueError(f"Unsupported sample_rate: {sample_rate}")


def build_model(args, logger):
    return ADINet(seq_len=args.window_size, logger=logger)


def build_optimizer(args, model):
    name = args.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def format_flops_macs(params, flops=None, macs=None):
    parts = [f"Params: {params} ({params / 1e6:.6f} M)"]
    if macs is not None:
        parts.append(f"Total MACs: {macs / 1e9:.6f} GMACs")
    if flops is not None:
        parts.append(f"Estimated FLOPs: {flops / 1e9:.6f} GFLOPs")
    return ", ".join(parts)


def estimate_dwt_macs(model, dummy_input):
    dwt = getattr(model, "dwt", None)
    if dwt is None or not hasattr(dwt, "h0"):
        return 0

    with torch.no_grad():
        yl, yhs = dwt(dummy_input.unsqueeze(1).float())

    filter_len = int(dwt.h0.shape[-1])
    output_elements = yl.numel() + sum(yh.numel() for yh in yhs)
    return int(output_elements * filter_len)


def log_model_complexity(model, args, logger):
    params = sum(p.numel() for p in model.parameters())
    if profile is None:
        logger.warning("thop is not installed; MAC/FLOP profiling is skipped.")
        logger.info("Model complexity: %s", format_flops_macs(params))
        return

    model_for_profile = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.randn(1, args.window_size)
    with torch.no_grad():
        try:
            base_macs, thop_params = profile(
                model_for_profile,
                inputs=(dummy_input,),
                verbose=False,
            )
            dwt_macs = estimate_dwt_macs(model_for_profile, dummy_input)
            total_macs = int(base_macs + dwt_macs)
            estimated_flops = int(2 * total_macs)
            logger.info(
                "Model complexity: %s, Base MACs: %.6f GMACs, DWT MACs: %.6f GMACs, THOP Params: %d",
                format_flops_macs(params, estimated_flops, total_macs),
                base_macs / 1e9,
                dwt_macs / 1e9,
                int(thop_params),
            )
        except Exception as exc:
            logger.warning("MAC/FLOP profiling failed: %s", exc)
            logger.info("Model complexity: %s", format_flops_macs(params))


def format_eval_results(results):
    parts = []
    for key, value in results.items():
        if key == "ConfusionMatrix":
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            parts.append(f"{key}: {value:.4f}")
    return ", ".join(parts)


def format_confusion_matrix(confusion_matrix):
    return "[[TN, FP], [FN, TP]] = [[{TN}, {FP}], [{FN}, {TP}]]".format(
        **confusion_matrix
    )


def log_experiment_header(args, logger, stage, data_type, data_paths, sample_rate):
    logger.info("--------------------------- Dataset ---------------------------")
    logger.info("%s dataset: %s", stage, data_type)
    logger.info("Config file: %s", args.config)
    logger.info("Sample rate: %s Hz", sample_rate)
    for data_path in data_paths:
        logger.info("Loading data from %s", data_path)

    logger.info("------------------------ Hyperparameters ----------------------")
    for key, value in sorted(vars(args).items()):
        logger.info("%s: %s", key, value)


def log_model_summary(model, args, logger):
    logger.info("--------------------------- Model -----------------------------")
    logger.info("Parameter count: %d", sum(p.numel() for p in model.parameters()))
    logger.info(model)
    logger.info("------------------------- Complexity --------------------------")
    log_model_complexity(model, args, logger)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
