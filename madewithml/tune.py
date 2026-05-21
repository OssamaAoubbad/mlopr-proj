import datetime
import itertools
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from madewithml import data, train, utils
from madewithml.config import logger, mlflow
from madewithml.models import SimpleModel

# Initialize Typer CLI app
app = typer.Typer()


def product_dicts(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


@app.command()
def tune_models(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    initial_params: Annotated[str, typer.Option(help="initial config for the tuning workload.")] = None,
    num_runs: Annotated[int, typer.Option(help="number of runs in this tuning experiment.")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save the tuning results to.")] = None,
) -> Dict[str, Any]:
    """Hyperparameter tuning experiment using grid search."""
    utils.set_seeds()

    train_loop_config = {
        "num_samples": num_samples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
    }

    df = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_df, val_df = data.stratify_split(df, stratify="tag", test_size=0.2)
    tags = sorted(train_df["tag"].unique())
    train_loop_config["num_classes"] = len(tags)

    preprocessor = data.CustomPreprocessor().fit(train_df)
    train_ds = preprocessor.transform(train_df)
    val_ds = preprocessor.transform(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)

    param_grid = json.loads(initial_params or "{}")
    if "train_loop_config" in param_grid:
        param_grid = param_grid["train_loop_config"]
    param_grid = {k: v if isinstance(v, list) else [v] for k, v in param_grid.items()}
    combinations = product_dicts(param_grid)

    results = []
    for index, params in enumerate(combinations):
        if index >= num_runs:
            break

        config = {**train_loop_config, **params}
        device = utils.get_device()
        model = SimpleModel(
            vocab_size=max(2, len(preprocessor.vocab)),
            embedding_dim=128,
            num_classes=config["num_classes"],
            dropout_p=config.get("dropout_p", 0.3),
        ).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("lr_factor", 0.8),
            patience=int(config.get("lr_patience", 3)),
        )

        run_metrics = []
        with mlflow.start_run(experiment_name=experiment_name) as run:
            mlflow.log_params(config)
            for epoch in range(config["num_epochs"]):
                train_loss = train.train_step(train_loader, batch_size, model, config["num_classes"], loss_fn, optimizer)
                val_loss, _, _ = train.eval_step(val_loader, batch_size, model, config["num_classes"], loss_fn)
                scheduler.step(val_loss)
                epoch_metrics = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]},
                    step=epoch,
                )
                run_metrics.append(epoch_metrics)

            with tempfile.TemporaryDirectory() as dp:
                model.save(dp)
                with open(Path(dp, "metadata.json"), "w") as fp:
                    json.dump({"class_to_index": preprocessor.class_to_index, "vocab": preprocessor.vocab}, fp)
                mlflow.pytorch.log_model(model, artifact_path="model")
                mlflow.log_artifact(str(Path(dp, "metadata.json")), artifact_path="model")

            results.append(
                {
                    "run_id": run.info.run_id,
                    "params": config,
                    "metrics": run_metrics,
                }
            )

    d = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "runs": results,
    }
    logger.info(json.dumps(d, indent=2))
    if results_fp:
        utils.save_dict(d, results_fp)
    return d


if __name__ == "__main__":
    app()
