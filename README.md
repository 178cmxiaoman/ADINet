# ADINet: Adaptive Dependency Interaction Network for Arc Fault Detection

This repository contains the official PyTorch implementation of **ADINet**, a lightweight arc fault detection model built around time-frequency feature interaction.

ADINet processes raw current windows with four stages:

1. **1D-DWT feature extraction** using a three-level Haar wavelet decomposition.
2. **Low-rank Feature Fusion (LFF)** to align and fuse multi-band wavelet features with low-rank fully connected blocks.
3. **Time-Frequency Feature Interaction (TFFI)** with Adaptive Dependency Interaction (ADI) blocks.
4. **Softmax classification** for binary normal/fault detection.

The implementation is cleaned for paper release and only keeps the ADINet training and evaluation path. Baseline model implementations and experimental artifacts are intentionally not included.

## Dataset

Experiments use the public IAED dataset:

> Intever public database for Arcing Event Detection, DOI: https://dx.doi.org/10.21227/j9eg-mz75

The paper evaluates six IAED splits:

| Split | Sampling rate | Window size | Samples |
| --- | ---: | ---: | ---: |
| HR-mix1 | 10 kHz | 1000 | 22,430 |
| HR-single | 10 kHz | 1000 | 11,526 |
| HR-all | 10 kHz | 1000 | 33,956 |
| LR-mix2 | 6.4 kHz | 1024 | 6,288 |
| LR-single | 6.4 kHz | 1024 | 14,084 |
| LR-all | 6.4 kHz | 1024 | 20,372 |

The code expects preprocessed CSV files organized as:

```text
IAED-processed/
  10000_multi_airconditioner02-refrigerator_m2/
    U.csv
    I.csv
    label.csv
  ...
```

Each YAML file under `yaml/data/` describes one subset member. Set `data.data_path` in `yaml/config/default.yaml` to the local `IAED-processed` directory. You do not need to edit the individual data YAML files.

## Installation

```bash
conda create -n adinet python=3.10 -y
conda activate adinet
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if you want GPU acceleration. See the official PyTorch installation page for the command that matches your CUDA version.

## Configuration

The default configuration is `yaml/config/default.yaml`, which is the HR-mix1 setup. Dataset-specific configurations are:

| Split | Config |
| --- | --- |
| HR-mix1 | `yaml/config/default.yaml` |
| HR-single | `yaml/config/HR-single.yaml` |
| HR-all | `yaml/config/HR-all.yaml` |
| LR-mix2 | `yaml/config/LR-mix2.yaml` |
| LR-single | `yaml/config/LR-single.yaml` |
| LR-all | `yaml/config/LR-all.yaml` |

Important fields:

| Field | Description |
| --- | --- |
| `data.data_type` | One of `HR-mix1`, `HR-single`, `HR-all`, `LR-mix2`, `LR-single`, `LR-all`, or `test`. |
| `data.data_path` | Local path to the processed IAED root directory. |
| `training.epochs` | Default is 300, matching the paper setup. |
| `training.batch_size` | Default is 2048. |
| `training.lr` | Default is 1e-3. |
| `training.optimizer` | Default is AdamW. |
| `system.save_path` | Directory for checkpoints and mirrored logs. |

Window size is inferred automatically: 1000 for 10 kHz HR data and 1024 for 6.4 kHz LR data.

## Training

```bash
python main.py --config yaml/config/default.yaml
```

The training script writes:

```text
checkpoints/ADINet-HR-mix1/
  best_model.pth
  last_model.pth
  model-<epoch>.pth
  Train-<split>-<timestamp>.log
```

`best_model.pth` is selected by validation accuracy on the 20% test split.

## Evaluation

```bash
python test.py --config yaml/config/default.yaml
```

By default, evaluation tests both `best_model.pth` and `last_model.pth` over five random 80/20 splits and reports mean +/- standard deviation for Accuracy, Precision, Recall, F1, FPR, FNR, ROC-AUC, PR-AUC, inference time, and the aggregated confusion matrix.

To evaluate a single checkpoint, set:

```yaml
evaluation:
  checkpoint_mode: "best"  # or "last"
  checkpoint_path: null
```

## Paper Settings

The paper uses:

| Hyperparameter | Value |
| --- | --- |
| Wavelet | Haar |
| DWT levels | 3 |
| ADI blocks | 3 |
| HR patch length | 20 |
| LR patch length | 32 |
| Epochs | 300 |
| Batch size | 2048 |
| Learning rate | 1e-3 |
| Optimizer | AdamW |
| Train/test split | 80% / 20% |

## Citation

If this code is useful for your work, please cite:

```bibtex
@article{adinet2026,
  title = {ADINet: An Adaptive Dependency Interaction Network for Arc Fault Detection},
  year = {2026},
  note = {Manuscript under review}
}
```

The final BibTeX entry will be updated after publication.
