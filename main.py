# 导入必要模块
import os
import copy
import torch
import logging
import argparse
import yaml
import numpy as np
import pandas as pd

import torch.nn as nn
from datetime import datetime
from collections import defaultdict

from eval import eval
from model.AE import AE
from model.CNN import CNN
from model.CNNTransformerParallel import CNNTransformerParallelNetwork
from model.MobileNetV2 import MobileNetV2
from model.Transformer import TransformerModel
from model.GRU import GRUModel
from model.MSCNN import MultiScaleCNN
from model.ADINet import ADINet  # AGDI 已内联到 ADINet.py

from torch.utils.data import DataLoader, TensorDataset
from utils.GetData import load_data_parallel, get_data
from utils.ML import ModelTrainer
from utils.preprocess import preprocess
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
    parser = argparse.ArgumentParser(description="Electrical Appliance Identification Training")
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

    if not hasattr(args, "pipeline"):
        args.pipeline = "deep_learning"

    if not hasattr(args, "n_runs"):
        args.n_runs = 5

    return args


######################深度学习训练######################
def train_deep_learning(I, label, sample_rate, args):
    logger = logging.getLogger(__name__)
    logger.info(f"使用 {args.device} 训练模型")

    if args.window_size is None:
        args.window_size = 1000 if sample_rate == 10000 else 1024 if sample_rate == 6400 else None
        if args.window_size is None:
            logger.error(f"不支持的 sample_rate: {sample_rate}")
            raise ValueError(f"不支持的 sample_rate: {sample_rate}")

    # 模型选择
    if args.model == "CNN":
        model = CNN(args.window_size)
    elif args.model == "MSCNN":
        model = MultiScaleCNN(args.window_size)
    elif args.model == "ADINet":
        model = ADINet(seq_len=args.window_size, logger=logger)
    elif args.model == "AE":
        model = AE(args.window_size)
    elif args.model == "MobileNetV2":
        model = MobileNetV2(args.window_size, num_classes=getattr(args, "num_classes", 2), width_mult=getattr(args, "width_mult", 1.0))
    elif args.model == "CNNTransformerParallel":
        model = CNNTransformerParallelNetwork(
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
        model = TransformerModel(args.window_size)
    elif args.model == "GRU":
        model = GRUModel(args.window_size)
    else:
        logger.error(f"不支持的 model_type: {args.model}")
        raise ValueError(f"不支持的 model_type: {args.model}")

    logger.info(f"使用 {sample_rate} 的数据集, 窗口大小为 {args.window_size}")

    logger.info("---------------------------超参数设置---------------------------")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("---------------------------模型结构---------------------------")
    logger.info("模型参数量为：{}".format(sum(p.numel() for p in model.parameters())))
    logger.info(model)
    logger.info("---------------------------模型复杂度---------------------------")
    log_model_complexity(model, args, logger)

    loss_fn = nn.CrossEntropyLoss()

    # 优化器选择
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        logger.error(f"不支持的 optimizer: {args.optimizer}")
        raise ValueError(f"不支持的 optimizer: {args.optimizer}")

    model.train()
    model = model.to(args.device)

    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(I.values, label)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    logger.info("---------------------------开始模型训练---------------------------")
    start_time = datetime.now()
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.to(args.device)
            batch_y = batch_y.to(args.device)

            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        del batch_X, batch_y

        # 每个 epoch 后做一次验证，用于 best_model 判断
        results = eval(model, test_dataloader, args.device)

        if (epoch + 1) % args.log_interval == 0:
            logger.info(f"Epoch [{epoch + 1}/{args.epochs}], " f"Loss: {loss_sum / len(train_dataloader):.4f}, " f"{format_eval_results(results)}")

        # 保存 best_model（按 Accuracy 指标）
        current_acc = results.get("Accuracy", None)
        if current_acc is not None and current_acc > best_accuracy:
            best_accuracy = current_acc
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pth")
            logger.info(f"Best model updated at epoch {epoch + 1}, Accuracy: {best_accuracy:.4f}")

        # 始终覆盖保存 last_model
        torch.save(model.state_dict(), f"{args.save_path}/last_model.pth")

        # 保留按 interval 的历史快照（兼容已有流程）
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model, f"{args.save_path}/model-{epoch + 1}.pth")

    # 兼容 test.py 默认 last 路径解析：model-{epochs}.pth
    torch.save(model, f"{args.save_path}/model-{args.epochs}.pth")

    end_time = datetime.now()
    logger.info("---------------------------结束模型训练---------------------------")
    logger.info(f"总共用时{(end_time - start_time)}，平均用时 {(end_time - start_time)/args.epochs} /epoch")

    inference_start_time = datetime.now()
    results = eval(model, test_dataloader, args.device)
    inference_end_time = datetime.now()
    inference_time = (inference_end_time - inference_start_time).total_seconds()
    logger.info(f"模型推理总用时：{inference_time:.4f} 秒")
    logger.info("测试结果为：")
    logger.info(format_eval_results(results))
    logger.info(f"混淆矩阵: {format_confusion_matrix(results['ConfusionMatrix'])}")


######################传统机器学习训练######################
def train_traditional(input_features, label, sample_rate, args):
    logger = logging.getLogger(__name__)
    logger.info(f"传统机器学习特征对应采样率: {sample_rate}")
    trainer = ModelTrainer()

    all_results = defaultdict(lambda: defaultdict(list))

    logger.info(f"开始进行 {args.n_runs} 次传统机器学习训练和测试")
    for run in range(args.n_runs):
        logger.info(f"第 {run + 1}/{args.n_runs} 次运行")

        split_data_tuple = trainer.split_data(input_features, label)
        results = trainer.train_and_evaluate(*split_data_tuple)

        for model_name, metrics in results.items():
            for metric_name, metric_value in metrics.items():
                all_results[model_name][metric_name].append(metric_value)

    model_names = list(all_results.keys())
    if not model_names:
        logger.warning("没有传统机器学习结果可显示")
        return

    metric_names = list(all_results[model_names[0]].keys())

    logger.info("---------------------------传统模型结果汇总（均值 ± 标准差，百分比）---------------------------")
    result_data = {}
    for metric_name in metric_names:
        result_data[metric_name] = {}
        for model_name in model_names:
            values = all_results[model_name][metric_name]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            result_data[metric_name][model_name] = f"{mean_val * 100:.2f} ± {std_val * 100:.2f}"

    result_df = pd.DataFrame(result_data).T
    logger.info(f"\n{result_df}")


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    data_type, data_paths = get_data(args.data_type)
    logger = setup_logger(task_type="Train", data_type=args.data_type, extra_log_dirs=[args.save_path])

    logger.info("---------------------------数据集描述---------------------------")
    logger.info("训练数据集为{}".format(data_type))
    logger.info(f"配置文件: {args.config}")
    logger.info(f"训练流程: {args.pipeline}")
    for data_path in data_paths:
        logger.info("Loading data from {}".format(data_path))

    start_time = datetime.now()
    U_combined, I_combined, label_combined, sample_rate = load_data_parallel(data_paths, args.data_path)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    logger.info(f"数据读取完成，总共用时{elapsed_time:.2f} 秒")

    if args.pipeline == "traditional":
        logger.info("---------------------------数据预处理---------------------------")
        features_input = preprocess(I_combined.values, sample_rate)
        logger.info(f"预处理完成，特征维度: {features_input.shape}")

        logger.info("---------------------------开始训练传统机器学习模型---------------------------")
        train_traditional(features_input, label_combined, sample_rate, args)
    elif args.pipeline == "deep_learning":
        train_deep_learning(I_combined, label_combined, sample_rate, args)
    else:
        raise ValueError(f"不支持的 pipeline: {args.pipeline}，可选: deep_learning / traditional")
