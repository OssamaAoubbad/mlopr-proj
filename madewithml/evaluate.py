import datetime
import json
from typing import Dict

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from madewithml import predict, utils
from madewithml.config import logger

# Initialize Typer CLI app
app = typer.Typer()


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Get overall performance metrics."""
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }


def get_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict) -> Dict:
    """Get per class performance metrics."""
    per_class_metrics = {}
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(class_to_index):
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    return dict(sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))


@slicing_function()
def nlp_llm(x):
    """NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame) -> Dict:
    """Get performance metrics for slices."""
    slice_metrics = {}
    frame = df.copy()
    frame["text"] = frame["title"].fillna("") + " " + frame["description"].fillna("")
    slices = PandasSFApplier([nlp_llm, short_text]).apply(frame)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if mask.sum():
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro")
            slice_metrics[slice_name] = {
                "precision": metrics[0],
                "recall": metrics[1],
                "f1": metrics[2],
                "num_samples": int(mask.sum()),
            }
    return slice_metrics


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    dataset_loc: Annotated[str, typer.Option(help="dataset (with labels) to evaluate on")] = None,
    results_fp: Annotated[str, typer.Option(help="location to save evaluation results to")] = None,
) -> Dict:
    """Evaluate on the holdout dataset."""
    df = pd.read_csv(dataset_loc)
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    preprocessed_ds = predictor.get_preprocessor().transform(df)
    loader = DataLoader(preprocessed_ds, batch_size=64, collate_fn=utils.collate_fn)
    y_true = preprocessed_ds.targets
    y_preds = []
    for batch in loader:
        y_preds.extend(predictor.predict(batch))
    y_pred = np.array(y_preds)

    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=predictor.get_preprocessor().class_to_index),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, df=df),
    }
    logger.info(json.dumps(metrics, indent=2))
    if results_fp:
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":
    app()
