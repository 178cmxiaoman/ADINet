import torch

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _round_metric(value):
    return round(float(value), 4)


def eval(model, test_dataloader, device):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():  # 在测试时不计算梯度
        for batch_X, batch_y in test_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            if outputs.ndim == 2 and outputs.size(1) > 1:
                positive_scores = torch.softmax(outputs, dim=1)[:, 1]
            else:
                positive_scores = torch.sigmoid(outputs.reshape(-1))

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(positive_scores.cpu().numpy())

    # 计算准确率
    accuracy = _safe_divide(correct, total)
    # print(f"Test Accuracy: {accuracy:.4f}")

    # 计算混淆矩阵，固定 labels 顺序为 [0, 1]，避免某一类缺失时 ravel 失败。
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    false_alarm_rate = _safe_divide(FP, FP + TN)
    missed_detection_rate = _safe_divide(FN, FN + TP)

    # 计算 Precision, Recall 和 F1 分数
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    if len(set(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # 返回结果
    results = {
        "Accuracy": _round_metric(accuracy),
        "Precision": _round_metric(precision),
        "Recall": _round_metric(recall),
        "F1": _round_metric(f1),
        "FalseAlarmRate": _round_metric(false_alarm_rate),
        "MissedDetectionRate": _round_metric(missed_detection_rate),
        "ROC-AUC": _round_metric(roc_auc),
        "PR-AUC": _round_metric(pr_auc),
        "ConfusionMatrix": {
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "TP": int(TP),
        },
    }
    return results
