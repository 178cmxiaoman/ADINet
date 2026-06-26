# 导入必要模块
import os
import time
import copy
import torch
import logging
import argparse
import yaml
import numpy as np
from collections import defaultdict
from datetime import datetime

from eval import eval
from model.AE import AE
from model.CNN import CNN
from model.CNNTransformerParallel import CNNTransformerParallelNetwork
from model.MobileNetV2 import MobileNetV2
from model.Transformer import TransformerModel
from model.GRU import GRUModel
from model.MSCNN import MultiScaleCNN
from model.ADINet import ADINet

from torch.utils.data import DataLoader, TensorDataset
from utils.GetData import load_data_parallel, get_data
from utils.ML import ModelTrainer
from utils.utils_logger import setup_logger

try:
    from thop import profile  # type: ignore[import-not-found]
except ImportError:
    profile = None


class Args:
    def __str__(self):
        return str(self.__dict__)


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
        logger.warning("未安装 thop，无法计算 MACs/FLOPs；请先安装 thop 后重试。")
        logger.info(f"模型复杂度：{format_flops_macs(params)}")
        return

    model_for_profile = copy.deepcopy(model).cpu().eval()
    dummy_input = torch.randn(1, args.window_size)
    with torch.no_grad():
        try:
            base_macs, thop_params = profile(model_for_profile, inputs=(dummy_input,), verbose=False)
            dwt_macs = estimate_dwt_macs(model_for_profile, dummy_input)
            total_macs = int(base_macs + dwt_macs)
            estimated_flops = int(2 * total_macs)
            logger.info("模型复杂度：" f"{format_flops_macs(params, estimated_flops, total_macs)}, " f"Base MACs: {base_macs / 1e9:.6f} GMACs, " f"DWT MACs: {dwt_macs / 1e9:.6f} GMACs, " f"THOP Params: {int(thop_params)}")
        except Exception as exc:
            logger.warning(f"MACs/FLOPs 计算失败: {exc}")
            logger.info(f"模型复杂度：{format_flops_macs(params)}")


def format_eval_results(results):
    parts = []
    for key, value in results.items():
        if key == "ConfusionMatrix":
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            parts.append(f"{key}: {value:.4f}")
    return ", ".join(parts)


def format_confusion_matrix(confusion_matrix):
    return "[[TN, FP], [FN, TP]] = " f"[[{confusion_matrix['TN']}, {confusion_matrix['FP']}], " f"[{confusion_matrix['FN']}, {confusion_matrix['TP']}]]"


def parse_args():
    parser = argparse.ArgumentParser(description="Electrical Appliance Identification Testing")
    parser.add_argument("--config", type=str, default="/root/Projects/ArcFaultDetection/yaml/config/Transformer.yaml", help="Path to config yaml file")
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

    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # 测试默认参数（兼容旧配置）
    if not hasattr(args, "n_runs"):
        args.n_runs = 5
    if not hasattr(args, "checkpoint_mode"):
        args.checkpoint_mode = "both"
    if not hasattr(args, "checkpoint_path"):
        # 若为空则根据 checkpoint_mode 自动拼接 save_path 下的文件名
        args.checkpoint_path = None
    if not hasattr(args, "model_save_format"):
        # "full_model" | "state_dict"
        args.model_save_format = "full_model"

    return args


def build_model(args, logger):
    if args.model == "CNN":
        return CNN(args.window_size)
    elif args.model == "MSCNN":
        return MultiScaleCNN(args.window_size)
    elif args.model == "ADINet":
        return ADINet(seq_len=args.window_size, logger=logger)
    elif args.model == "AE":
        return AE(args.window_size)
    elif args.model == "MobileNetV2":
        return MobileNetV2(args.window_size, num_classes=getattr(args, "num_classes", 2), width_mult=getattr(args, "width_mult", 1.0))
    elif args.model == "CNNTransformerParallel":
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
    elif args.model == "Transformer":
        return TransformerModel(args.window_size)
    elif args.model == "GRU":
        return GRUModel(args.window_size)
    else:
        raise ValueError(f"不支持的 model_type: {args.model}")


def resolve_checkpoint_path(args, model_type="last"):
    if args.checkpoint_path:
        return args.checkpoint_path

    if model_type == "best":
        return os.path.join(args.save_path, "best_model.pth")

    # last 优先使用 state_dict 形式的 last_model.pth；不存在则回退到历史 full_model 命名
    last_state_dict_path = os.path.join(args.save_path, "last_model.pth")
    if os.path.exists(last_state_dict_path):
        return last_state_dict_path

    return os.path.join(args.save_path, f"model-{args.epochs}.pth")


def load_test_model(args, logger, model_type="last"):
    ckpt_path = resolve_checkpoint_path(args, model_type=model_type)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")

    logger.info(f"加载模型: {ckpt_path}")

    # 先按配置加载；若失败则自动回退（state_dict <-> full_model）
    if args.model_save_format == "state_dict":
        try:
            test_model = build_model(args, logger).to(args.device)
            state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            test_model.load_state_dict(state_dict)
        except Exception:
            logger.warning("按 state_dict 加载失败，尝试按完整模型加载")
            test_model = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            test_model = test_model.to(args.device)
    else:
        try:
            test_model = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            test_model = test_model.to(args.device)
        except Exception:
            logger.warning("按完整模型加载失败，尝试按 state_dict 加载")
            test_model = build_model(args, logger).to(args.device)
            state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            test_model.load_state_dict(state_dict)

    test_model.eval()
    return test_model


def evaluate_one_model(I, label, args, test_model, model_name):
    logger = logging.getLogger(__name__)
    all_results = defaultdict(list)
    all_confusion_matrices = []
    all_test_time = []
    trainer = ModelTrainer()

    logger.info(f"---------------------------开始{args.n_runs}次{model_name}测试---------------------------")
    for run in range(args.n_runs):
        logger.info(f"========== {model_name} 第 {run + 1}/{args.n_runs} 次测试 ==========")

        X_train, X_test, y_train, y_test = trainer.split_data(I.values, label)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        logger.info("---------------------------开始测试模型---------------------------")
        start_time = datetime.now()
        results = eval(test_model, test_dataloader, args.device)
        end_time = datetime.now()

        test_time = (end_time - start_time).total_seconds()
        per_sample_time = test_time / len(test_dataloader.dataset) if len(test_dataloader.dataset) > 0 else 0.0

        logger.info(f"{model_name} 第 {run + 1} 次测试结果: {format_eval_results(results)}")
        logger.info(f"{model_name} 第 {run + 1} 次混淆矩阵: {format_confusion_matrix(results['ConfusionMatrix'])}")
        logger.info(f"{model_name} 第 {run + 1} 次测试完成，总共用时 {end_time - start_time}")
        logger.info(f"{model_name} 第 {run + 1} 次平均每个样本测试时间: {per_sample_time * 1000:.6f} ms")

        all_test_time.append(per_sample_time)
        for key, value in results.items():
            if key == "ConfusionMatrix":
                all_confusion_matrices.append(value)
                continue
            all_results[key].append(value)

    logger.info(f"---------------------------{model_name}统计结果（均值 ± 标准差）---------------------------")
    final_results = {}
    for key, values in all_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
        final_results[key] = {"mean": mean_val, "std": std_val, "values": values}
        logger.info(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

    if all_test_time:
        mean_time = np.mean(all_test_time)
        std_time = np.std(all_test_time, ddof=1) if len(all_test_time) > 1 else 0.0
        final_results["InferenceTime"] = {"mean": mean_time, "std": std_time, "values": all_test_time}
        logger.info(f"InferenceTime: {mean_time * 1000:.6f} ± {std_time * 1000:.6f} ms/sample")

    if all_confusion_matrices:
        summed_confusion_matrix = {"TN": int(sum(cm["TN"] for cm in all_confusion_matrices)), "FP": int(sum(cm["FP"] for cm in all_confusion_matrices)), "FN": int(sum(cm["FN"] for cm in all_confusion_matrices)), "TP": int(sum(cm["TP"] for cm in all_confusion_matrices))}
        final_results["ConfusionMatrix"] = {"sum": summed_confusion_matrix, "values": all_confusion_matrices}
        logger.info(f"AggregatedConfusionMatrix: {format_confusion_matrix(summed_confusion_matrix)}")

    paper_format = {}
    for key, val in final_results.items():
        if key == "ConfusionMatrix":
            continue
        if key == "InferenceTime":
            paper_format[key] = f"{val['mean'] * 1000:.4f} ± {val['std'] * 1000:.4f}"
        else:
            paper_format[key] = f"{val['mean'] * 100:.2f} ± {val['std'] * 100:.2f}"

    logger.info(f"{model_name} 论文格式结果: {paper_format}")
    return final_results, paper_format


def test(I, label, sample_rate, args):
    logger = logging.getLogger(__name__)
    logger.info(f"使用 {args.device} 测试模型")

    if args.window_size is None:
        args.window_size = 1000 if sample_rate == 10000 else 1024 if sample_rate == 6400 else None
        if args.window_size is None:
            logger.error(f"不支持的 sample_rate: {sample_rate}")
            raise ValueError(f"不支持的 sample_rate: {sample_rate}")

    model = build_model(args, logger)

    logger.info(f"使用 {sample_rate} 的数据集, 窗口大小为 {args.window_size}")
    logger.info("---------------------------超参数设置---------------------------")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("---------------------------模型结构---------------------------")
    logger.info("模型参数量为：{}".format(sum(p.numel() for p in model.parameters())))
    logger.info(model)
    logger.info("---------------------------模型复杂度---------------------------")
    log_model_complexity(model, args, logger)

    if args.checkpoint_mode == "both":
        best_model = load_test_model(args, logger, model_type="best")
        best_results, best_paper_format = evaluate_one_model(I, label, args, best_model, "best_model")

        last_model = load_test_model(args, logger, model_type="last")
        last_results, last_paper_format = evaluate_one_model(I, label, args, last_model, "last_model")

        logger.info("=" * 80)
        logger.info("两个模型结果汇总（论文格式，百分比）")
        logger.info(f"best_model: {best_paper_format}")
        logger.info(f"last_model: {last_paper_format}")
        logger.info("=" * 80)

        return {"best": {"results": best_results, "paper_format": best_paper_format}, "last": {"results": last_results, "paper_format": last_paper_format}}

    model_type = "best" if args.checkpoint_mode == "best" else "last"
    test_model = load_test_model(args, logger, model_type=model_type)
    final_results, paper_format = evaluate_one_model(I, label, args, test_model, f"{model_type}_model")
    return {model_type: {"results": final_results, "paper_format": paper_format}}


if __name__ == "__main__":
    args = parse_args()

    data_type, data_paths = get_data(args.data_type)
    logger = setup_logger(task_type="Test", data_type=args.data_type, extra_log_dirs=[args.save_path])

    logger.info("---------------------------数据集描述---------------------------")
    logger.info("测试数据集为{}".format(data_type))
    logger.info(f"配置文件: {args.config}")
    for data_path in data_paths:
        logger.info("Loading data from {}".format(data_path))

    start_time = datetime.now()
    U_combined, I_combined, label_combined, sample_rate = load_data_parallel(data_paths, args.data_path)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    logger.info(f"数据读取完成，总共用时{elapsed_time:.2f} 秒")

    test(I_combined, label_combined, sample_rate, args)
