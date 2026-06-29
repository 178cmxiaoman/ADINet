from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from eval import eval
from utils.GetData import get_data, load_data_parallel
from utils.experiment import (
    build_model,
    build_optimizer,
    ensure_output_dir,
    format_confusion_matrix,
    format_eval_results,
    infer_window_size,
    load_args,
    log_experiment_header,
    log_model_summary,
    set_random_seed,
)
from utils.utils_logger import setup_logger


def make_dataloaders(I, label, args):
    X_train, X_test, y_train, y_test = train_test_split(
        I.values,
        label,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=label,
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, test_loader


def train_adinet(I, label, sample_rate, args, logger):
    args.window_size = infer_window_size(args, sample_rate)
    model = build_model(args, logger).to(args.device)

    logger.info("Using %s for training.", args.device)
    log_model_summary(model, args, logger)

    train_loader, test_loader = make_dataloaders(I, label, args)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

    logger.info("-------------------------- Training ---------------------------")
    start_time = datetime.now()
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(args.device)
            batch_y = batch_y.to(args.device)

            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        results = eval(model, test_loader, args.device)
        mean_loss = loss_sum / max(len(train_loader), 1)

        if (epoch + 1) % args.log_interval == 0:
            logger.info(
                "Epoch [%d/%d], Loss: %.4f, %s",
                epoch + 1,
                args.epochs,
                mean_loss,
                format_eval_results(results),
            )

        current_acc = results.get("Accuracy")
        if current_acc is not None and current_acc > best_accuracy:
            best_accuracy = current_acc
            torch.save(model.state_dict(), f"{args.save_path}/best_model.pth")
            logger.info(
                "Best model updated at epoch %d, Accuracy: %.4f",
                epoch + 1,
                best_accuracy,
            )

        torch.save(model.state_dict(), f"{args.save_path}/last_model.pth")
        if args.save_interval and (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.save_path}/model-{epoch + 1}.pth")

    end_time = datetime.now()
    logger.info("------------------------- Finished ----------------------------")
    logger.info("Total training time: %s", end_time - start_time)
    logger.info("Average time per epoch: %s", (end_time - start_time) / args.epochs)

    results = eval(model, test_loader, args.device)
    logger.info("Final validation results: %s", format_eval_results(results))
    logger.info("Final confusion matrix: %s", format_confusion_matrix(results["ConfusionMatrix"]))


def main():
    args = load_args("ADINet training")
    set_random_seed(args.seed)
    ensure_output_dir(args.save_path)

    data_type, data_paths = get_data(args.data_type)
    logger = setup_logger(
        task_type="Train",
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
    log_experiment_header(args, logger, "Training", data_type, data_paths, sample_rate)
    logger.info("Data loaded in %.2f seconds.", elapsed_time)
    logger.info(
        "Class distribution: %s",
        {int(k): int(v) for k, v in zip(*np.unique(label_combined, return_counts=True))},
    )

    train_adinet(I_combined, label_combined, sample_rate, args, logger)


if __name__ == "__main__":
    main()
