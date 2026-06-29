import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml


DATASETS = {
    "HR-mix1": [
        "yaml/data/00-10000_multi_airconditioner02-refrigerator_m2.yaml",
        "yaml/data/01-10000_multi_kettle01-microwaveoven_m2.yaml",
        "yaml/data/02-10000_multi_refrigerator-airconditioner_m2.yaml",
        "yaml/data/03-10000_multi_ricecooker-electricoven_m2.yaml",
        "yaml/data/04-10000_multi_inductioncooker-airpurifier_m2.yaml",
        "yaml/data/05-10000_multi_airpurifier-airconditioner03_m2.yaml",
        "yaml/data/06-10000_multi_microwaveoven-waterheater_m2.yaml",
        "yaml/data/07-10000_multi_electricoven-washingmachine_m2.yaml",
        "yaml/data/08-10000_multi_airconditioner01-inductioncooker_m2.yaml",
    ],
    "HR-single": [
        "yaml/data/09-10000_single_inductioncooker_m1.yaml",
        "yaml/data/10-10000_single_refrigerator_m1.yaml",
        "yaml/data/11-10000_single_electricoven_m1.yaml",
        "yaml/data/12-10000_single_ricecooker_m1.yaml",
        "yaml/data/13-10000_single_kettle01_m1.yaml",
        "yaml/data/14-10000_single_microwaveoven_m1.yaml",
    ],
    "HR-all": [
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
    ],
    "LR-mix2": [
        "yaml/data/15-6400_multi_microwaveoven-refrigerator_m3.yaml",
        "yaml/data/16-6400_multi_electricoven-refrigerator_m3.yaml",
        "yaml/data/17-6400_multi_kettle02-airpurifier_m3.yaml",
    ],
    "LR-single": [
        "yaml/data/18-6400_single_inductioncooker_m1.yaml",
        "yaml/data/19-6400_single_refrigerator_m1.yaml",
        "yaml/data/20-6400_single_kettle02_m1.yaml",
        "yaml/data/21-6400_single_electricoven_m1.yaml",
        "yaml/data/22-6400_single_ricecooker_m1.yaml",
        "yaml/data/23-6400_single_kettle01_m1.yaml",
        "yaml/data/24-6400_single_microwaveoven_m1.yaml",
    ],
    "LR-all": [
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
    ],
    "test": ["yaml/data/test.yaml"],
}


def _resolve_data_path(original_path, data_root):
    if not data_root:
        return original_path

    sample_dir = os.path.basename(os.path.dirname(original_path))
    file_name = os.path.basename(original_path)
    return os.path.join(data_root, sample_dir, file_name)


def load_single_file(data_config_path, data_root=None):
    with open(data_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    u_path = _resolve_data_path(data_config["U_path"], data_root)
    i_path = _resolve_data_path(data_config["I_path"], data_root)
    label_path = _resolve_data_path(data_config["label_path"], data_root)

    U = pd.read_csv(u_path)
    I = pd.read_csv(i_path)
    label = pd.read_csv(label_path)["label"].values.astype(np.int64)
    window_size = int(data_config["window_size"])
    sample_rate = int(data_config["sample_rate"])
    return U, I, label, window_size, sample_rate


def load_data_parallel(data_paths, data_root=None):
    U_combined = pd.DataFrame()
    I_combined = pd.DataFrame()
    label_combined = np.array([], dtype=np.int64)
    sample_rates = set()
    window_sizes = set()

    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda path: load_single_file(path, data_root=data_root),
                data_paths,
            )
        )

    for U, I, label, window_size, sample_rate in results:
        U_combined = pd.concat([U_combined, U], ignore_index=True)
        I_combined = pd.concat([I_combined, I], ignore_index=True)
        label_combined = np.concatenate([label_combined, label])
        sample_rates.add(sample_rate)
        window_sizes.add(window_size)

    if len(sample_rates) != 1 or len(window_sizes) != 1:
        raise ValueError(
            "A single run must use one sample rate and one window size. "
            f"Got sample_rates={sorted(sample_rates)}, window_sizes={sorted(window_sizes)}."
        )

    return U_combined, I_combined, label_combined, sample_rates.pop()


def get_data(idx_data):
    if idx_data not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown data_type '{idx_data}'. Valid values: {valid}")
    return idx_data, DATASETS[idx_data]
