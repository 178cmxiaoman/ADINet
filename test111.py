# 导入必要模块
import os
from datetime import datetime

import torch
import logging
import yaml

from eval import eval
from torch.utils.data import DataLoader, TensorDataset
from utils.GetData import load_data_parallel, get_data
from utils.ML import ModelTrainer
from utils.utils_logger import setup_logger
from model.ADINet import ADINet
from model.MobileNetV2 import MobileNetV2

from model.CNNTransformerParallel import CNNTransformerParallelNetwork

""" 10个低频, 15个高频
00-10000_multi_airconditioner02-refrigerator_m2.yaml
01-10000_multi_kettle01-microwaveoven_m2.yaml
02-10000_multi_refrigerator-airconditioner_m2.yaml
03-10000_multi_ricecooker-electricoven_m2.yaml
04-10000_multi_inductioncooker-airpurifier_m2.yaml
05-10000_multi_airpurifier-airconditioner03_m2.yaml
06-10000_multi_microwaveoven-waterheater_m2.yaml
07-10000_multi_electricoven-washingmachine_m2.yaml
08-10000_multi_airconditioner01-inductioncooker_m2.yaml
09-10000_single_inductioncooker_m1.yaml
10-10000_single_refrigerator_m1.yaml
11-10000_single_electricoven_m1.yaml
12-10000_single_ricecooker_m1.yaml
13-10000_single_kettle01_m1.yaml
14-10000_single_microwaveoven_m1.yaml

15-6400_multi_microwaveoven-refrigerator_m3.yaml
16-6400_multi_electricoven-refrigerator_m3.yaml
17-6400_multi_kettle02-airpurifier_m3.yaml
18-6400_single_inductioncooker_m1.yaml
19-6400_single_refrigerator_m1.yaml
20-6400_single_kettle02_m1.yaml
21-6400_single_electricoven_m1.yaml
22-6400_single_ricecooker_m1.yaml
23-6400_single_kettle01_m1.yaml
24-6400_single_microwaveoven_m1.yaml
"""

logger = setup_logger()
window_size = 1000


def resolve_data_base_path():
    """Resolve the dataset root path for YAML files.

    YAML files may contain Windows-style absolute paths. This helper maps them to
    the current environment so the same configs work on WSL/Linux too.
    """
    default_candidates = ["/home/xiaoman/Projects/Datasets/", "/home/xiaoman/Projects/AFCI/Datasets/"]

    config_path = "/home/xiaoman/Projects/AFCI/ArcFaultDetection/yaml/config/CNNTransformerParallel-HR-mix1.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        configured_path = config_dict.get("data", {}).get("data_path")
        if configured_path and os.path.exists(configured_path):
            return configured_path

    for candidate in default_candidates:
        if os.path.exists(candidate):
            return candidate

    return default_candidates[0]


######################训练######################
def load_mobilenetv2_model(model_path, window_size, device):
    """Load a MobileNetV2 checkpoint that may be a full model or state_dict."""
    logger.info(f"加载模型: {model_path}")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.Module):
            return model.to(device).eval()
        raise TypeError("loaded object is not a full model")
    except Exception:
        logger.warning("按完整模型加载失败，尝试按 state_dict 加载")
        model = MobileNetV2(window_size, num_classes=2)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def load_cnn_transformer_parallel_model(model_path, window_size, device):
    """Load a CNNTransformerParallel checkpoint that may be a full model or state_dict."""
    logger.info(f"加载模型: {model_path}")
    model_kwargs = dict(num_classes=2, cnn_channels=(64, 256), d_model=192, n_head=4, n_layers=4, dim_feedforward=384, dropout=0.1)
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(model, torch.nn.Module):
            return model.to(device).eval()
        raise TypeError("loaded object is not a full model")
    except Exception:
        logger.warning("按完整模型加载失败，尝试按 state_dict 加载")
        model = CNNTransformerParallelNetwork(window_size, **model_kwargs)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def test(I, label, sample_rate, data_base_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用 {device} 训练模型")

    # 设置窗口
    if sample_rate == 10000:
        window_size = 1000
    elif sample_rate == 6400:
        window_size = 1024
    else:
        raise ValueError("Sample rate not supported")
    # model_path = "/home/xiaoman/Projects/AFCI/checkpoints/model-1.pth"  # 替换为实际的模型路径
    # model_path = "/home/xiaoman/Projects/AFCI/checkpoints/master/MobileNetV2/HR-mix1/best_model.pth"  # MobileNetV2
    model_path = "/home/xiaoman/Projects/AFCI/checkpoints/master/CNNTransformerParallel/HR-mix1/best_model.pth"  # CNNTransformerParallel

    # 加载模型
    # model = TransformerModel(window_size)  # 初始化模型结构
    # model = ADINet(seq_len=window_size, logger=logger)
    model = load_cnn_transformer_parallel_model(model_path, window_size, device)  # CNNTransformerParallel

    logger.info(f"使用 {sample_rate} 的数据集, 窗口大小为 {window_size}")
    logger.info("---------------------------模型结构---------------------------")
    # logger.info(model)

    # 加载数据集
    trainer = ModelTrainer()

    # 数据预处理
    X_train, X_test, y_train, y_test = trainer.split_data(I.values, label)  # 得到X_train,X_test,y_train,y_test

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=True)

    logger.info("---------------------------开始测试模型---------------------------")
    start_time = datetime.now()

    # 使用 eval 函数对模型进行测试
    results = eval(model, test_dataloader, device)

    # 输出测试结果
    logger.info("测试结果为：")
    for key, value in results.items():
        logger.info(f"{key}: {value:.4f}")

    end_time = datetime.now()
    logger.info(f"测试完成，总共用时 {end_time - start_time}")


if __name__ == "__main__":
    # "HR-mix1", "HR-single","LR-mix2","LR-single","HR-all","LR-all","test"
    data_type, data_paths = get_data("HR-mix1")
    data_base_path = resolve_data_base_path()

    logger.info("训练数据集为{}".format(data_type))
    logger.info(f"数据根目录: {data_base_path}")
    for data_path in data_paths:
        logger.info("Loading data from {}".format(data_path))
    # 数据读取
    logger.info("---------------------------数据读取---------------------------")
    start_time = datetime.now()
    U_combined, I_combined, label_combined, sample_rate = load_data_parallel(data_paths, data_base_path)
    end_time = datetime.now()
    logger.info(f"数据读取完成，总共用时{end_time - start_time}")

    # 训练模型
    test(I_combined, label_combined, sample_rate, data_base_path)
