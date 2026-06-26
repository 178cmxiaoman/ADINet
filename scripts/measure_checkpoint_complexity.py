import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

try:
    from thop import profile
except ImportError as exc:
    raise SystemExit("Please install thop first: pip install thop") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
AFCI_ROOT = PROJECT_ROOT.parent

from model.ADINet import ADINet
from model.CNN import CNN
from model.GRU import GRUModel
from model.MSCNN import MultiScaleCNN
from model.MobileNetV2 import MobileNetV2
from model.Transformer import TransformerModel
from model.CNNTransformerParallel import CNNTransformerParallelNetwork


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def normalize_model_name(path):
    name = Path(path).stem.lower()
    aliases = {
        "adinet": "ADINet",
        "arcnet": "CNN",
        "cnn": "CNN",
        "gru": "GRU",
        "transformer": "Transformer",
        "mscnn": "MSCNN",
        "ms-cnn": "MSCNN",
        "mobilenet": "MobileNetV2",
        "mobilenetv2": "MobileNetV2",
        "ctpn": "CNNTransformerParallel",
        "ctpnn": "CNNTransformerParallel",
        "cnntransformerparallel": "CNNTransformerParallel",
    }
    compact_name = name.replace("_", "").replace("-", "")
    if compact_name in aliases:
        return aliases[compact_name]
    if name in aliases:
        return aliases[name]
    raise ValueError(f"Cannot infer model type from checkpoint name: {Path(path).name}")


def build_model(model_name, window_size):
    if model_name == "ADINet":
        return ADINet(seq_len=window_size, logger=NullLogger())
    if model_name == "CNN":
        return CNN(window_size)
    if model_name == "GRU":
        return GRUModel(window_size)
    if model_name == "Transformer":
        return TransformerModel(window_size)
    if model_name == "MSCNN":
        return MultiScaleCNN(window_size)
    if model_name == "MobileNetV2":
        return MobileNetV2(window_size)
    if model_name == "CNNTransformerParallel":
        return CNNTransformerParallelNetwork(window_size)
    raise ValueError(f"Unsupported model type: {model_name}")


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return clean_state_dict(value)

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return clean_state_dict(checkpoint)

    return None


def infer_window_size_from_model(model):
    if hasattr(model, "seq_len"):
        return int(model.seq_len)
    if hasattr(model, "input_length"):
        return int(model.input_length)
    if hasattr(model, "flatten_dim"):
        flatten_dim = int(model.flatten_dim)
        if model.__class__.__name__ == "CNN":
            return flatten_dim // 50
        if hasattr(model, "d_model"):
            return flatten_dim // int(model.d_model)
    if hasattr(model, "hidden_size") and hasattr(model, "fc"):
        first_linear = next((layer for layer in model.fc if isinstance(layer, nn.Linear)), None)
        if first_linear is not None:
            return int(first_linear.in_features) // int(model.hidden_size)
    return None


def load_model_from_checkpoint(path, window_size):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        inferred_window_size = infer_window_size_from_model(model)
        if inferred_window_size is not None:
            window_size = inferred_window_size
        return model.cpu().eval(), window_size, "full_model"

    model_name = normalize_model_name(path)
    model = build_model(model_name, window_size)
    state_dict = extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    model.load_state_dict(state_dict)
    return model.cpu().eval(), window_size, "state_dict"


def estimate_dwt_macs(model, dummy_input):
    dwt = getattr(model, "dwt", None)
    if dwt is None or not hasattr(dwt, "h0"):
        return 0

    with torch.no_grad():
        yl, yhs = dwt(dummy_input.unsqueeze(1).float())

    filter_len = int(dwt.h0.shape[-1])
    output_elements = yl.numel() + sum(yh.numel() for yh in yhs)
    return int(output_elements * filter_len)


def measure_checkpoint(path, window_size):
    model, window_size, checkpoint_type = load_model_from_checkpoint(path, window_size)
    dummy_input = torch.randn(1, window_size)

    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        base_macs, thop_params = profile(model, inputs=(dummy_input,), verbose=False)
        dwt_macs = estimate_dwt_macs(model, dummy_input)

    total_macs = int(base_macs + dwt_macs)
    return {
        "checkpoint": str(path),
        "model": normalize_model_name(path),
        "checkpoint_type": checkpoint_type,
        "window_size": int(window_size),
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
    print("| Checkpoint | Model | Type | Window | Params (M) | Base GMACs | DWT GMACs | Total GMACs | Est. GFLOPs |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {checkpoint} | {model} | {checkpoint_type} | {window_size} | {params_m:.6f} | "
            "{base_gmacs:.6f} | {dwt_gmacs:.6f} | {total_gmacs:.6f} | {gflops:.6f} |".format(
                checkpoint=Path(row["checkpoint"]).name,
                model=row["model"],
                checkpoint_type=row["checkpoint_type"],
                window_size=row["window_size"],
                params_m=row["params"] / 1e6,
                base_gmacs=row["base_macs"] / 1e9,
                dwt_gmacs=row["dwt_macs"] / 1e9,
                total_gmacs=row["total_macs"] / 1e9,
                gflops=row["estimated_flops"] / 1e9,
            )
        )


def expand_paths(paths):
    expanded = []
    for path_text in paths:
        path = resolve_path(path_text)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.pth")))
        else:
            expanded.append(path)
    return expanded


def resolve_path(path_text):
    path = Path(path_text).expanduser()
    if path.exists():
        return path

    mac_prefix = Path("/Users/zhenyuzhou/Projects/PC-WSL/Projects/AFCI")
    path_string = str(path)
    prefix_string = str(mac_prefix)
    if path_string.startswith(prefix_string):
        mapped = AFCI_ROOT / path_string[len(prefix_string) :].lstrip("/")
        if mapped.exists():
            return mapped
        path = mapped

    stripped = Path(str(path).rstrip())
    if stripped.exists():
        return stripped

    with_space = Path(str(stripped) + " ")
    if with_space.exists():
        return with_space

    parent = stripped.parent
    if parent.exists():
        matches = sorted(parent.glob(stripped.name + "*"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Path not found: {path_text}\n"
        f"Current project root: {PROJECT_ROOT}\n"
        f"If you are running on the remote server, pass a Linux path such as: "
        f"{AFCI_ROOT / 'checkpoints' / 'ComputationalComplexityComparison'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Measure Params, GMACs, and estimated GFLOPs from checkpoint paths.")
    parser.add_argument("paths", default="/home/xiaoman/Projects/AFCI/checkpoints/ComputationalComplexityComparison/ADINet.pth", nargs="+", help="Checkpoint .pth file(s) or directory path(s).")
    parser.add_argument("--window-size", type=int, default=1000, help="Fallback input length for state_dict checkpoints.")
    args = parser.parse_args()

    checkpoint_paths = expand_paths(args.paths)
    if not checkpoint_paths:
        raise SystemExit("No checkpoint files found.")

    rows = []
    for path in checkpoint_paths:
        rows.append(measure_checkpoint(path, args.window_size))
    print_markdown(rows)


if __name__ == "__main__":
    main()
