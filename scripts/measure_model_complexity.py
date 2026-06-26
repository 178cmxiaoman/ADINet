import argparse
import copy
from pathlib import Path
import sys

import torch

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Please install PyYAML first: pip install pyyaml") from exc

try:
    from thop import profile
except ImportError as exc:
    raise SystemExit("Please install thop first: pip install thop") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.AE import AE
from model.ADINet import ADINet
from model.CNN import CNN
from model.CNNTransformerParallel import CNNTransformerParallelNetwork
from model.GRU import GRUModel
from model.MSCNN import MultiScaleCNN
from model.MobileNetV2 import MobileNetV2
from model.Transformer import TransformerModel


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class Args:
    pass


def load_args(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    args = Args()
    for category, params in config_dict.items():
        if isinstance(params, dict):
            for key, value in params.items():
                setattr(args, key, value)
        else:
            setattr(args, category, params)

    if getattr(args, "window_size", None) is None:
        data_type = getattr(args, "data_type", "")
        if data_type.startswith("HR"):
            args.window_size = 1000
        elif data_type.startswith("LR"):
            args.window_size = 1024
        else:
            raise ValueError(f"Cannot infer window_size from data_type={data_type!r}")

    return args


def build_model(args):
    logger = NullLogger()
    if args.model == "CNN":
        return CNN(args.window_size)
    if args.model == "MSCNN":
        return MultiScaleCNN(args.window_size)
    if args.model == "ADINet":
        return ADINet(seq_len=args.window_size, logger=logger)
    if args.model == "AE":
        return AE(args.window_size)
    if args.model == "MobileNetV2":
        return MobileNetV2(
            args.window_size,
            num_classes=getattr(args, "num_classes", 2),
            width_mult=getattr(args, "width_mult", 1.0),
        )
    if args.model == "CNNTransformerParallel":
        return CNNTransformerParallelNetwork(
            args.window_size,
            num_classes=getattr(args, "num_classes", 2),
            cnn_channels=tuple(getattr(args, "cnn_channels", [64, 256])),
            d_model=getattr(args, "d_model", 192),
            n_head=getattr(args, "n_head", 4),
            n_layers=getattr(args, "n_layers", 4),
            dim_feedforward=getattr(args, "dim_feedforward", 384),
            dropout=getattr(args, "dropout", 0.1),
        )
    if args.model == "Transformer":
        return TransformerModel(args.window_size)
    if args.model == "GRU":
        return GRUModel(args.window_size)
    raise ValueError(f"Unsupported model: {args.model}")


def estimate_dwt_macs(model, dummy_input):
    dwt = getattr(model, "dwt", None)
    if dwt is None or not hasattr(dwt, "h0"):
        return 0

    with torch.no_grad():
        yl, yhs = dwt(dummy_input.unsqueeze(1).float())

    filter_len = int(dwt.h0.shape[-1])
    output_elements = yl.numel() + sum(yh.numel() for yh in yhs)
    return int(output_elements * filter_len)


def measure_one(config_path):
    args = load_args(config_path)
    model = build_model(args).cpu().eval()
    model_for_profile = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.randn(1, args.window_size)

    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        base_macs, thop_params = profile(model_for_profile, inputs=(dummy_input,), verbose=False)
        dwt_macs = estimate_dwt_macs(model_for_profile, dummy_input)

    total_macs = int(base_macs + dwt_macs)
    return {
        "config": str(config_path),
        "model": args.model,
        "data_type": getattr(args, "data_type", ""),
        "window_size": args.window_size,
        "params": int(params),
        "thop_params": int(thop_params),
        "base_macs": int(base_macs),
        "dwt_macs": int(dwt_macs),
        "total_macs": total_macs,
        "estimated_flops": int(2 * total_macs),
    }


def print_markdown(rows):
    print("THOP reports MACs. Estimated FLOPs are reported as 2 x Total MACs.")
    print()
    print("| Config | Model | Data | Window | Params (M) | Base GMACs | DWT GMACs | Total GMACs | Est. GFLOPs |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {config} | {model} | {data_type} | {window_size} | {params_m:.6f} | "
            "{base_gmacs:.6f} | {dwt_gmacs:.6f} | {total_gmacs:.6f} | {gflops:.6f} |".format(
                config=Path(row["config"]).name,
                model=row["model"],
                data_type=row["data_type"],
                window_size=row["window_size"],
                params_m=row["params"] / 1e6,
                base_gmacs=row["base_macs"] / 1e9,
                dwt_gmacs=row["dwt_macs"] / 1e9,
                total_gmacs=row["total_macs"] / 1e9,
                gflops=row["estimated_flops"] / 1e9,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Measure model Params, GMACs, and estimated GFLOPs.")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML config path(s).")
    args = parser.parse_args()

    rows = [measure_one(Path(config_path)) for config_path in args.configs]
    print_markdown(rows)


if __name__ == "__main__":
    main()
