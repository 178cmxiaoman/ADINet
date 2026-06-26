# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 14:47
# @Author  : TianHuiyun
# @Email   : tianhuiyun@intelever.com
# @File    : GetData.py

import os
import glob
import math
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from scipy.fftpack import fft, ifft
import csv
import yaml
from concurrent.futures import ThreadPoolExecutor


class DataProcess:
    def __init__(self, path, window_size):
        """
        初始化函数，设置路径和窗口大小，并加载相关文件。
        :param path: 数据文件所在的路径
        :param window_size: 时间窗口大小
        """
        self.path = path
        self.window_size = int(window_size)
        # 获取时间文件名（以 "time" 开头的文件）
        self.time_filename = [i for i in os.listdir(self.path) if i.startswith("time")][0]
        # 提取时间格式（文件名中 "_" 后的部分）
        self.time_format = self.time_filename.split("_")[-1].split(".")[0]
        # 构建时间文件的完整路径
        self.time_path = os.path.join(path, self.time_filename)
        # 获取所有 UI 数据文件的路径列表（以 "UI.txt" 结尾的文件）
        self.filename_list = glob.glob(path + "/*UI.txt")

    # get UI original data from a single file
    def get_data_origin(self, file_path):
        """
        从单个文件中读取原始的 UI 数据。
        :param file_path: UI 数据文件的路径
        :return: 包含电压 (U) 和电流 (I) 的 DataFrame
        """
        # 读取文件，分隔符为空格，无表头，跳过错误行，不进行引用处理
        # data = pd.read_csv(file_path, header=None, sep=" ",error_bad_lines=False,low_memory=False,quoting=csv.QUOTE_NONE)
        data = pd.read_csv(file_path, header=None, sep=" ", on_bad_lines="warn", low_memory=False, quoting=csv.QUOTE_NONE)
        # 只保留前两列（电压和电流）
        data = data.iloc[:, :2]
        # 设置列名
        data.columns = ["U", "I"]
        # 将数据转换为数值类型，无法转换的设置为 NaN
        data = data.apply(pd.to_numeric, errors="coerce", downcast="float")
        # 删除包含 NaN 的行
        data = data.dropna(how="any", axis=0)
        print("数据路径:", file_path)
        print("数据shape:", data.shape)
        return data

    def get_time_data(self):
        """
        读取时间标注数据，并根据时间格式进行处理。
        :return: 处理后的时间数据 DataFrame
        """
        # 读取时间文件，分隔符为空格，无表头
        time_data = pd.read_csv(self.time_path, header=None, sep=" ")
        if self.time_format == "format1":
            # 如果时间格式为 format1，设置列名并计算相对时间
            time_data.columns = ["start", "load", "arc", "extinction", "end"]
            time_data["diff"] = time_data["end"] - time_data["start"]
            time_data["start_copy"] = time_data["start"]
            time_data["start"] = (time_data["load"] - time_data["start_copy"]) / time_data["diff"]
            time_data["load1"] = (time_data["arc"] - time_data["start_copy"]) / time_data["diff"]
            time_data["arc"] = (time_data["extinction"] - time_data["start_copy"]) / time_data["diff"]
            time_data["load2"] = (time_data["end"] - time_data["start_copy"]) / time_data["diff"]
            time_data = time_data[["start", "load1", "arc", "load2"]]
        elif self.time_format == "format2":
            # 如果时间格式为 format2，设置列名并计算相对时间
            """
            start   负载启动时间比例。
            load1   第一次电弧发生时间比例。
            arc1    第一次电弧熄灭时间比例。
            load2   第二次电弧发生时间比例。
            arc2    第二次电弧熄灭时间比例。
            load3   负载关闭时间比例。
            end     实验结束时间比例。
            """
            time_data.columns = ["start", "load", "arc1", "extinction1", "arc2", "extinction2", "load_off", "end"]
            time_data["diff"] = time_data["end"] - time_data["start"]
            time_data["start_copy"] = time_data["start"]
            time_data["start"] = (time_data["load"] - time_data["start_copy"]) / time_data["diff"]
            time_data["load1"] = (time_data["arc1"] - time_data["start_copy"]) / time_data["diff"]
            time_data["arc1"] = (time_data["extinction1"] - time_data["start_copy"]) / time_data["diff"]
            time_data["load2"] = (time_data["arc2"] - time_data["start_copy"]) / time_data["diff"]
            time_data["arc2"] = (time_data["extinction2"] - time_data["start_copy"]) / time_data["diff"]
            time_data["load3"] = (time_data["load_off"] - time_data["start_copy"]) / time_data["diff"]
            time_data["end"] = (time_data["end"] - time_data["start_copy"]) / time_data["diff"]
            time_data = time_data[["start", "load1", "arc1", "load2", "arc2", "load3", "end"]]

        return time_data

    def make_label_multi(self, id):
        """
        为单个文件生成多类别标签(load, arc 等）。
        :param id: 文件在列表中的索引
        :return: 包含标签的 DataFrame
        """
        # 根据 id 获取文件路径
        file_path = self.filename_list[id]
        # 读取 UI 数据
        data = self.get_data_origin(file_path)
        # 获取对应的时间数据
        time = self.get_time_data().iloc[id]  # read time info of this id
        size = data.shape[0]
        data["label"] = np.NaN
        if self.time_format == "format1":
            # 根据时间格式计算各阶段的起始和结束位置
            num_start = math.floor(size * time["start"])
            num_load1 = math.floor(size * time["load1"])
            num_arc = math.floor(size * time["arc"])
            # 为不同阶段设置标签
            data.iloc[num_start:num_load1, -1] = "load"
            data.iloc[num_load1:num_arc, -1] = "arc"
            data.iloc[num_arc:, -1] = "load"
        elif self.time_format == "format2":
            # 根据时间格式计算各阶段的起始和结束位置
            num_start = math.floor(size * time["start"])
            num_load1 = math.floor(size * time["load1"])
            num_arc1 = math.floor(size * time["arc1"])
            num_load2 = math.floor(size * time["load2"])
            num_arc2 = math.floor(size * time["arc2"])
            num_load3 = math.floor(size * time["load3"])
            # 为不同阶段设置标签
            data.iloc[:num_start, -1] = "start"
            data.iloc[num_start:num_load1, -1] = "load"
            data.iloc[num_load1:num_arc1, -1] = "arc"
            data.iloc[num_arc1:num_load2, -1] = "load"
            data.iloc[num_load2:num_arc2, -1] = "arc"
            data.iloc[num_arc2:num_load3, -1] = "load"
            data.iloc[num_load3:, -1] = "end"
        # 只保留标签为 "load" 或 "arc" 的数据
        return data[(data["label"] == "load") | (data["label"] == "arc")]

    def make_label_binary(self, id):
        """
        为单个文件生成二分类标签（arc: 1, normal: 0）。
        :param id: 文件在列表中的索引
        :return: 包含二分类标签的 DataFrame
        """
        # 获取多类别标签数据
        data = self.make_label_multi(id)
        # 将 "load" 标签转换为 0，"arc" 标签转换为 1
        data.loc[data["label"] == "load", "label"] = 0
        data.loc[data["label"] == "arc", "label"] = 1
        return data

    def build_sample_data_head(self):
        """
        构建样本数据的表头（特征列名）。
        :return: 包含特征列名的列表
        """
        column_list = []
        # 添加电压特征列名
        for i in range(self.window_size):
            column_list.append("U_%d" % (i + 1))
        # 添加电流特征列名
        for j in range(self.window_size, self.window_size * 2):
            column_list.append("I_%d" % (j - self.window_size + 1))
        # 去重并排序
        keys = list(set(column_list))
        keys.sort(key=column_list.index)
        # 添加标签列名
        keys = keys + ["label"]
        return keys

    # build sample set according to time window

    def build_sample(self, id, task="train"):
        """
        根据时间窗口构建样本集。
        :param id: 文件在列表中的索引
        :param task: 任务类型（"train" 或 "save"）
        :return: 包含样本的 DataFrame
        """
        ws = self.window_size
        # 获取二分类标签数据
        data = self.make_label_binary(id)
        size = data.shape[0] // ws
        # 获取表头
        keys = self.build_sample_data_head()
        df = None
        for i in range(size):
            # 截取一个时间窗口的数据
            df_sub = data.iloc[i * ws : (i + 1) * ws]
            if df_sub["label"].unique().size == 1:
                # 构建样本特征值
                values = df_sub["U"].tolist() + df_sub["I"].tolist() + df_sub["label"].unique().tolist()
                if task == "train":
                    # 对训练任务进行过滤
                    if len([i for i in values[-(self.window_size + 1) : -1] if abs(i) <= 2.0]) >= 0.8 * ws:
                        pass
                    else:
                        # 构建样本字典
                        dict_ = {k: v for k, v in itertools.zip_longest(keys, values)}
                        # 将样本添加到 DataFrame
                        df = pd.concat([df, pd.DataFrame(dict_, index=[i])])

                elif task == "save":
                    # 对保存任务直接构建样本
                    dict_ = {k: v for k, v in itertools.zip_longest(keys, values)}
                    df = pd.concat([df, pd.DataFrame(dict_, index=[i])])
        return df

    def merge_all_sample(self, task="train", is_save=False):
        """
        合并所有文件的样本数据。
        :param task: 任务类型（"train" 或 "save"）
        :param is_save: 是否保存结果到文件
        :return: 合并后的 DataFrame
        """
        df = None
        for i in range(len(self.filename_list)):
            # 获取单个文件的样本数据
            data_one = self.build_sample(i, task=task)
            # 合并样本数据
            df = pd.concat([df, data_one], axis=0)
        # 重置索引
        df = df.reset_index(drop=True)
        # 从路径中提取信息并添加到 DataFrame
        name_list = self.path.split("/")
        df["sample_rate"] = name_list[-3].split("_")[0]
        df["sensor"] = name_list[-3].split("_")[1]
        df["load_config"] = name_list[-2]
        df["device_name"] = name_list[-1].split("_")[0]
        df["connection_method"] = name_list[-1].split("_")[1]
        print(df.shape)
        if is_save:
            # 根据任务类型保存数据
            if task == "save":
                df.to_csv(r"/home/zzy/AFCI/IAED-master/Arc-fault-detection-main/data/data_after_process/recorder/{}.csv".format(self.path.split("/")[-1]))

            elif task == "train":
                df.to_csv(r"/home/zzy/AFCI/IAED-master/Arc-fault-detection-main/data/data_for_train/recorder/{}.csv".format(self.path.split("/")[-1]))
            return df
        else:
            return df


def load_single_file(data_path, new_base_path=None):
    """
    加载单个 YAML 文件中的数据。
    """
    with open(data_path, "r") as file:
        config = yaml.safe_load(file)
    # 替换路径中的基础部分
    if new_base_path:
        config["data"]["U_path"] = config["data"]["U_path"].replace("E:/AFCI/Datasets/", new_base_path)
        config["data"]["I_path"] = config["data"]["I_path"].replace("E:/AFCI/Datasets/", new_base_path)
        config["data"]["label_path"] = config["data"]["label_path"].replace("E:/AFCI/Datasets/", new_base_path)

    U = pd.read_csv(config["data"]["U_path"])
    I = pd.read_csv(config["data"]["I_path"])
    label = pd.read_csv(config["data"]["label_path"])["label"].values
    window_size = config["data"]["window_size"]
    sample_rate = config["data"]["sample_rate"]
    return U, I, label, window_size, sample_rate


def load_data_parallel(data_paths, new_base_path=None):
    """
    使用多线程并行加载多个文件的数据。
    """
    U_combined = pd.DataFrame()
    I_combined = pd.DataFrame()
    label_combined = np.array([])

    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(load_single_file, data_paths))

    with ThreadPoolExecutor() as executor:
        # 使用 lambda 将 new_base_path 传递给 load_single_file
        results = list(executor.map(lambda path: load_single_file(path, new_base_path=new_base_path), data_paths))

    for U, I, label, _, sample_rate in results:
        U_combined = pd.concat([U_combined, U], ignore_index=True)
        I_combined = pd.concat([I_combined, I], ignore_index=True)
        label_combined = np.concatenate([label_combined, label])

    return U_combined, I_combined, label_combined, sample_rate


def get_data(idx_data):
    """
    根据数据集索引加载对应的数据集路径和类型。

    参数:
        idx_data (str): 数据集的索引名称，用于指定加载哪种数据集。
                        可选值包括:
                        - "HR-mix1": 高采样率混合负载数据集 1
                        - "HR-single": 高采样率单负载数据集
                        - "LR-mix1": 低采样率混合负载数据集 1
                        - "LR-single": 低采样率单负载数据集
                        - "HR-all": 高采样率所有负载数据集
                        - "LR-all": 低采样率所有负载数据集
                        - "test": 测试数据集

    返回:
        tuple: 包含以下两个元素的元组:
            - data_type (str): 数据集的类型名称（与 idx_data 对应）。
            - data_paths (list): 数据集对应的 YAML 配置文件路径列表。

    示例:
        data_type, data_paths = get_data("HR-mix1")
        print(data_type)  # 输出: "HR-mix1"
        print(data_paths)  # 输出: ["yaml/data/00-10000_multi_airconditioner02-refrigerator_m2.yaml", ...]
    """
    if idx_data == "HR-mix1":
        # HR-mix1
        # Sample amout = 22430
        # Time duration = 2243.0 sec
        data_type = "HR-mix1"
        data_paths = [
            "yaml/data/00-10000_multi_airconditioner02-refrigerator_m2.yaml",
            "yaml/data/01-10000_multi_kettle01-microwaveoven_m2.yaml",
            "yaml/data/02-10000_multi_refrigerator-airconditioner_m2.yaml",
            "yaml/data/03-10000_multi_ricecooker-electricoven_m2.yaml",
            "yaml/data/04-10000_multi_inductioncooker-airpurifier_m2.yaml",
            "yaml/data/05-10000_multi_airpurifier-airconditioner03_m2.yaml",
            "yaml/data/06-10000_multi_microwaveoven-waterheater_m2.yaml",
            "yaml/data/07-10000_multi_electricoven-washingmachine_m2.yaml",
            "yaml/data/08-10000_multi_airconditioner01-inductioncooker_m2.yaml",
        ]

    elif idx_data == "HR-single":
        # HR-single
        # Sample amout = 11526
        # Time duration = 1152.6 sec
        data_type = "HR-single"
        data_paths = [
            "yaml/data/09-10000_single_inductioncooker_m1.yaml",
            "yaml/data/10-10000_single_refrigerator_m1.yaml",
            "yaml/data/11-10000_single_electricoven_m1.yaml",
            "yaml/data/12-10000_single_ricecooker_m1.yaml",
            "yaml/data/13-10000_single_kettle01_m1.yaml",
            "yaml/data/14-10000_single_microwaveoven_m1.yaml",
        ]

    elif idx_data == "LR-mix2":
        # LR-mix2
        # Sample amout = 6288
        # Time duration = 1006.08 sec
        data_type = "LR-mix2"
        data_paths = [
            "yaml/data/15-6400_multi_microwaveoven-refrigerator_m3.yaml",
            "yaml/data/16-6400_multi_electricoven-refrigerator_m3.yaml",
            "yaml/data/17-6400_multi_kettle02-airpurifier_m3.yaml",
        ]

    elif idx_data == "LR-single":
        #  LR-single
        # Sample amout = 14084
        # Time duration = 2253.44 sec
        data_type = "LR-single"
        data_paths = [
            "yaml/data/18-6400_single_inductioncooker_m1.yaml",
            "yaml/data/19-6400_single_refrigerator_m1.yaml",
            "yaml/data/20-6400_single_kettle02_m1.yaml",
            "yaml/data/21-6400_single_electricoven_m1.yaml",
            "yaml/data/22-6400_single_ricecooker_m1.yaml",
            "yaml/data/23-6400_single_kettle01_m1.yaml",
            "yaml/data/24-6400_single_microwaveoven_m1.yaml",
        ]

    elif idx_data == "HR-all":
        #  HR-all
        # Sample amout = 33956
        # Time duration = 3395.6 sec
        data_type = "HR-all"
        data_paths = [
            "yaml/data/00-10000_multi_airconditioner02-refrigerator_m2.yaml",
            "yaml/data/01-10000_multi_kettle01-microwaveoven_m2.yaml",
            "yaml/data/02-10000_multi_refrigerator-airconditioner_m2.yaml",
            "yaml/data/03-10000_multi_ricecooker-electricoven_m2.yaml",
            "yaml/data/04-10000_multi_inductioncooker-airpurifier_m2.yaml",
            "yaml/data/05-10000_multi_airpurifier-airconditioner03_m2.yaml",
            "yaml/data/06-10000_multi_microwaveoven-waterheater_m2.yaml",
            "yaml/data/07-10000_multi_electricoven-washingmachine_m2.yaml",
            "yaml/data/08-10000_multi_airconditioner01-inductioncooker_m2.yaml",
            "yaml/data/09-10000_single_inductioncooker_m1.yaml",
            "yaml/data/10-10000_single_refrigerator_m1.yaml",
            "yaml/data/11-10000_single_electricoven_m1.yaml",
            "yaml/data/12-10000_single_ricecooker_m1.yaml",
            "yaml/data/13-10000_single_kettle01_m1.yaml",
            "yaml/data/14-10000_single_microwaveoven_m1.yaml",
        ]
    elif idx_data == "LR-all":
        # LR-all
        # Sample amout = 20372
        # Time duration = 3259.52 sec
        data_type = "LR-all"
        data_paths = [
            "yaml/data/15-6400_multi_microwaveoven-refrigerator_m3.yaml",
            "yaml/data/16-6400_multi_electricoven-refrigerator_m3.yaml",
            "yaml/data/17-6400_multi_kettle02-airpurifier_m3.yaml",
            "yaml/data/18-6400_single_inductioncooker_m1.yaml",
            "yaml/data/19-6400_single_refrigerator_m1.yaml",
            "yaml/data/20-6400_single_kettle02_m1.yaml",
            "yaml/data/21-6400_single_electricoven_m1.yaml",
            "yaml/data/22-6400_single_ricecooker_m1.yaml",
            "yaml/data/23-6400_single_kettle01_m1.yaml",
            "yaml/data/24-6400_single_microwaveoven_m1.yaml",
        ]
    elif idx_data == "test":
        data_type = "test"
        data_paths = ["yaml/data/test.yaml"]  # 测试数据集

    return data_type, data_paths
