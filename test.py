from collections import defaultdict
from datetime import datetime
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from eval import eval
from utils.GetData import get_data, load_data_parallel
from utils.experiment import (
    build_model,
    format_confusion_matrix,
    format_eval_results,
    infer_window_size,
    load_args,
    log_experiment_header,
    log_model_summary,
    set_random_seed,
)
from utils.utils_logger import setup_logger


def resolve_checkpoint_path(args, checkpoint_kind="last"):
    if args.checkpoint_path:
        return args.checkpoint_path
    if checkpoint_kind == "best":
        return os.path.join(args.save_path, "best_model.pth")
    return os.path.join(args.save_path, "last_model.pth")


def torch_load(path, device, weights_only):
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(args, logger, checkpoint_kind="last"):
    checkpoint_path = resolve_checkpoint_path(args, checkpoint_kind)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = build_model(args, logger).to(args.device)

    try:
        state_dict = torch_load(checkpoint_path, args.device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception:
        logger.warning("State-dict loading failed; trying a full-model checkpoint.")
        model = torch_load(checkpoint_path, args.device, weights_only=False)
        model = model.to(args.device)

    model.eval()
    return model


def make_test_loader(I, label, args, run):
    _, X_test, _, y_test = train_test_split(
        I.values,
        label,
        test_size=args.test_size,
        random_state=args.seed + run,
        stratify=label,
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def evaluate_checkpoint(I, label, args, model, checkpoint_name):
    logger = logging.getLogger(__name__)
    all_results = defaultdict(list)
    all_confusion_matrices = []
    all_inference_time = []

    logger.info("---------------------- %s evaluation ----------------------", checkpoint_name)
    for run in range(args.n_runs):
        test_loader = make_test_loader(I, label, args, run)

        start_time = datetime.now()
        results = eval(model, test_loader, args.device)
        elapsed = (datetime.now() - start_time).total_seconds()
        per_sample_time = elapsed / len(test_loader.dataset) if len(test_loader.dataset) else 0.0

        logger.info(
            "%s run %d/%d: %s",
            checkpoint_name,
            run + 1,
            args.n_runs,
            format_eval_results(results),
        )
        logger.info(
            "%s run %d/%d confusion matrix: %s",
            checkpoint_name,
            run + 1,
            args.n_runs,
            format_confusion_matrix(results["ConfusionMatrix"]),
        )
        logger.info(
            "%s run %d/%d inference time: %.6f ms/sample",
            checkpoint_name,
            run + 1,
            args.n_runs,
            per_sample_time * 1000,
        )

        all_inference_time.append(per_sample_time)
        for key, value in results.items():
            if key == "ConfusionMatrix":
                all_confusion_matrices.append(value)
            else:
                all_results[key].append(value)

    final_results = {}
    paper_format = {}
    logger.info("---------------------- %s summary -------------------------", checkpoint_name)
    for key, values in all_results.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        final_results[key] = {"mean": mean_val, "std": std_val, "values": values}
        paper_format[key] = f"{mean_val * 100:.2f} +/- {std_val * 100:.2f}"
        logger.info("%s: %.4f +/- %.4f", key, mean_val, std_val)

    if all_inference_time:
        mean_time = float(np.mean(all_inference_time))
        std_time = float(np.std(all_inference_time, ddof=1)) if len(all_inference_time) > 1 else 0.0
        final_results["InferenceTime"] = {
            "mean": mean_time,
            "std": std_time,
            "values": all_inference_time,
        }
        paper_format["InferenceTime"] = f"{mean_time * 1000:.4f} +/- {std_time * 1000:.4f}"
        logger.info("InferenceTime: %.6f +/- %.6f ms/sample", mean_time * 1000, std_time * 1000)

    if all_confusion_matrices:
        summed = {
            "TN": int(sum(cm["TN"] for cm in all_confusion_matrices)),
            "FP": int(sum(cm["FP"] for cm in all_confusion_matrices)),
            "FN": int(sum(cm["FN"] for cm in all_confusion_matrices)),
            "TP": int(sum(cm["TP"] for cm in all_confusion_matrices)),
        }
        final_results["ConfusionMatrix"] = {
            "sum": summed,
            "values": all_confusion_matrices,
        }
        logger.info("Aggregated confusion matrix: %s", format_confusion_matrix(summed))

    logger.info("%s paper-format results: %s", checkpoint_name, paper_format)
    return final_results, paper_format


def run_test(I, label, sample_rate, args, logger):
    args.window_size = infer_window_size(args, sample_rate)
    model_for_summary = build_model(args, logger).to(args.device)
    logger.info("Using %s for testing.", args.device)
    log_model_summary(model_for_summary, args, logger)

    valid_modes = {"best", "last", "both"}
    if args.checkpoint_mode not in valid_modes:
        raise ValueError(f"checkpoint_mode must be one of {sorted(valid_modes)}.")

    checkpoint_kinds = ["best", "last"] if args.checkpoint_mode == "both" else [args.checkpoint_mode]
    summaries = {}
    for checkpoint_kind in checkpoint_kinds:
        model = load_checkpoint(args, logger, checkpoint_kind)
        summaries[checkpoint_kind] = evaluate_checkpoint(
            I,
            label,
            args,
            model,
            f"{checkpoint_kind}_model",
        )

    logger.info("====================== Paper-format summary ======================")
    for checkpoint_kind, (_, paper_format) in summaries.items():
        logger.info("%s: %s", checkpoint_kind, paper_format)
    return summaries


def main():
    args = load_args("ADINet testing")
    set_random_seed(args.seed)

    data_type, data_paths = get_data(args.data_type)
    logger = setup_logger(
        task_type="Test",
        data_type=data_type,
        extra_log_dirs=[args.save_path],
    )

    start_time = datetime.now()
    _, I_combined, label_combined, sample_rate = load_data_parallel(
        data_paths,
        args.data_path,
    )
    elapsed_time = (datetime.now() - start_time).total_seconds()

    args.window_size = infer_window_size(args, sample_rate)
    log_experiment_header(args, logger, "Test", data_type, data_paths, sample_rate)
    logger.info("Data loaded in %.2f seconds.", elapsed_time)
    run_test(I_combined, label_combined, sample_rate, args, logger)


if __name__ == "__main__":
    main()
