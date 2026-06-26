# 导入必要模块
import torch
import tqdm
import pandas as pd
import torch.nn as nn

from eval import eval
from utils.GetData import DataProcess
from utils.ML import ModelTrainer
from model.CNN import CNN1D

import os

from torch.utils.data import DataLoader, TensorDataset

"""
10个低频, 15个高频
0-0 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/airconditioner02-refrigerator_m2 window_size: 1000
0-1 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/kettle01-microwaveoven_m2 window_size: 1000
0-2 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/refrigerator-airconditioner_m2 window_size: 1000
0-3 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/ricecooker-electricoven_m2 window_size: 1000
0-4 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/inductioncooker-airpurifier_m2 window_size: 1000
0-5 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/airpurifier-airconditioner03_m2 window_size: 1000
0-6 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/microwaveoven-waterheater_m2 window_size: 1000
0-7 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/electricoven-washingmachine_m2 window_size: 1000
0-8 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/multi-load/airconditioner01-inductioncooker_m2 window_size: 1000
0-9 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/inductioncooker_m1 window_size: 1000
0-10 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/refrigerator_m1 window_size: 1000
0-11 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/electricoven_m1 window_size: 1000
0-12 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/ricecooker_m1 window_size: 1000
0-13 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/kettle01_m1 window_size: 1000
0-14 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/10000_recorder/single-load/microwaveoven_m1 window_size: 1000
0-15 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/multi-load/microwaveoven-refrigerator_m3 window_size: 1024
0-16 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/multi-load/electricoven-refrigerator_m3 window_size: 1024
0-17 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/multi-load/kettle02-airpurifier_m3 window_size: 1024
0-18 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/inductioncooker_m1 window_size: 1024
0-19 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/refrigerator_m1 window_size: 1024
0-20 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/kettle02_m1 window_size: 1024
0-21 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/electricoven_m1 window_size: 1024
0-22 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/ricecooker_m1 window_size: 1024
0-23 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/kettle01_m1 window_size: 1024
24 : E:/AFCI/Datasets/IAED-master/InteverArcingDataSet-release1.0/6400_embedded/single-load/microwaveoven_m1 window_size: 1024
"""


def export_to_csv(U, I, label, output_dir="output"):
    """
    将 U, I 和 label 导出为 CSV 文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    U.to_csv(os.path.join(output_dir, "U.csv"), index=False)
    I.to_csv(os.path.join(output_dir, "I.csv"), index=False)
    pd.DataFrame(label, columns=["label"]).to_csv(
        os.path.join(output_dir, "label.csv"), index=False
    )
    print(f"数据已导出到 {output_dir} 目录")


def load_from_csv(input_dir="output"):
    """
    从 CSV 文件中加载 U, I 和 label。
    """
    U = pd.read_csv(os.path.join(input_dir, "U.csv"))
    I = pd.read_csv(os.path.join(input_dir, "I.csv"))
    label = pd.read_csv(os.path.join(input_dir, "label.csv"))["label"].values
    print(f"数据已从 {input_dir} 目录加载")
    return U, I, label


if __name__ == "__main__":
    # 数据集选择

    file_path = f"/root/Projects/Datasets/IAED-processed"
    window_size = 1024

    # 格式转换
    origin_df = DataProcess(file_path, window_size).merge_all_sample(is_save=False)
    sample_rate = origin_df["sample_rate"][0]
    print("数据加载完成")

    ######################特征提取######################
    # 原始数据电流时序图
    U = origin_df.iloc[:, 0:window_size]
    I = origin_df.iloc[:, window_size : window_size * 2]
    label = origin_df["label"].values

    export_to_csv(U, I, label)
