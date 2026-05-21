import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from madewithml import data, utils
from madewithml.config import EFS_DIR, MLFLOW_TRACKING_URI, logger, mlflow
from madewithml.models import SimpleModel

# Initialize Typer CLI app
app = typer.Typer()


def train_step(
    ds: DataLoader,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:  # pragma: no cover, tested via train workload
    """Train step."""
    model.train()
    loss = 0.0
    for i, batch in enumerate(ds):
        optimizer.zero_grad()
        z = model(batch)
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
        J = loss_fn(z, targets)
        J.backward()
        optimizer.step()
        loss += (J.detach().item() - loss) / (i + 1)
    return loss


def eval_step(
    ds: DataLoader,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
) -> tuple[float, np.ndarray, np.ndarray]:  # pragma: no cover, tested via train workload
    """Eval step."""
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(ds):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.array(y_trues), np.array(y_preds)


def make_data_loader(dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=utils.collate_fn)


@app.command()
def train_model(
    experiment_name: str = "mlops-project",
    dataset_loc: str = "datasets/dataset.csv",
    train_loop_config: str = '{"dropout_p":0.3,"lr":1e-5,"lr_factor":0.8,"lr_patience":3}',
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_samples: int = 100,
    num_epochs: int = 10,
    batch_size: int = 8,
    results_fp: str = "results.json",
) -> Dict:
    """Main train function for our PyTorch model."""
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    utils.set_seeds()

    df = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_df, val_df = data.stratify_split(df, stratify="tag", test_size=0.2)
    tags = sorted(train_df["tag"].unique())
    train_loop_config["num_classes"] = len(tags)

    preprocessor = data.CustomPreprocessor().fit(train_df)
    train_ds = preprocessor.transform(train_df)
    val_ds = preprocessor.transform(val_df)
    train_loader = make_data_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_data_loader(val_ds, batch_size=batch_size, shuffle=False)

    device = utils.get_device()
    model = SimpleModel(
        vocab_size=max(2, len(preprocessor.vocab)),
        embedding_dim=128,
        num_classes=train_loop_config["num_classes"],
        dropout_p=train_loop_config.get("dropout_p", 0.3),
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_loop_config.get("lr", 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_loop_config.get("lr_factor", 0.8),
        patience=int(train_loop_config.get("lr_patience", 3)),
    )

    metrics = []
    with mlflow.start_run(experiment_name=experiment_name) as run:
        mlflow.log_params(train_loop_config)
        for epoch in range(train_loop_config["num_epochs"]):
            train_loss = train_step(train_loader, batch_size, model, train_loop_config["num_classes"], loss_fn, optimizer)
            val_loss, _, _ = eval_step(val_loader, batch_size, model, train_loop_config["num_classes"], loss_fn)
            scheduler.step(val_loss)
            epoch_metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            metrics.append(epoch_metrics)

        with tempfile.TemporaryDirectory() as dp:
            model.save(dp)
            with open(Path(dp, "metadata.json"), "w") as metadata_fp:
                json.dump({"class_to_index": preprocessor.class_to_index, "vocab": preprocessor.vocab}, metadata_fp)
            mlflow.pytorch.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(Path(dp, "metadata.json")), artifact_path="model")

        run_id = run.info.run_id

    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "params": train_loop_config,
        "metrics": metrics,
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:
        utils.save_dict(d, results_fp)
    return d


if __name__ == "__main__":
    app()
