import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
from urllib.request import url2pathname

import mlflow
import numpy as np
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from madewithml.config import logger
from madewithml.data import CustomPreprocessor
from madewithml.models import SimpleModel
from madewithml.utils import collate_fn

# Initialize Typer CLI app
app = typer.Typer()


def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    d = {}
    for i, item in enumerate(prob):
        d[index_to_class[i]] = item
    return d


class TorchPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch):
        results = self.model.predict(collate_fn(batch))
        return {"output": results}

    def predict_proba(self, batch):
        results = self.model.predict_proba(collate_fn(batch))
        return {"output": results}

    def get_preprocessor(self):
        return self.preprocessor

    @classmethod
    def from_checkpoint(cls, artifact_dir: Path):
        metadata_path = artifact_dir / "model" / "metadata.json"
        with open(metadata_path, "r") as fp:
            metadata = json.load(fp)
        preprocessor = CustomPreprocessor(class_to_index=metadata["class_to_index"], vocab=metadata.get("vocab", {}))
        model = mlflow.pytorch.load_model(str(artifact_dir / "model"))
        return cls(preprocessor=preprocessor, model=model)


def predict_proba(ds: pd.DataFrame, predictor: TorchPredictor) -> List:
    preprocessed_ds = predictor.get_preprocessor().transform(ds)
    loader = DataLoader(preprocessed_ds, batch_size=32, collate_fn=collate_fn)
    results = []
    for batch in loader:
        probs = predictor.predict_proba(batch)["output"]
        for prob in probs:
            tag = predictor.get_preprocessor().index_to_class[int(np.argmax(prob))]
            results.append({"prediction": tag, "probabilities": format_prob(prob, predictor.get_preprocessor().index_to_class)})
    return results


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    run_id = sorted_runs.iloc[0].run_id
    print(run_id)
    return run_id


def get_best_checkpoint(run_id: str) -> Path:
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    parsed_uri = urlparse(artifact_uri)
    if parsed_uri.scheme == "file":
        artifact_dir = Path(url2pathname(parsed_uri.netloc + parsed_uri.path))
    else:
        artifact_dir = Path(artifact_uri)
    return artifact_dir


@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    title: Annotated[str, typer.Option(help="project title")] = None,
    description: Annotated[str, typer.Option(help="project description")] = None,
) -> Dict:
    sample_df = pd.DataFrame([
        {"title": title or "", "description": description or "", "tag": "other"}
    ])
    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    results = predict_proba(ds=sample_df, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    app()
