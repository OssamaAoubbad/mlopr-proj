import argparse
import json
import os
from http import HTTPStatus
from typing import Dict

import ray
import uvicorn
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from madewithml import evaluate, predict
from madewithml.config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


async def get_prediction_input(request: Request, title: str = "", description: str = "") -> Dict:
    """Get prediction inputs from a JSON body or query parameters."""
    data = {}
    body = await request.body()
    if body:
        data = json.loads(body)
    return {
        "title": data.get("title", title) or "",
        "description": data.get("description", description) or "",
    }


def create_app(run_id: str, threshold: float = 0.9) -> FastAPI:
    """Create a local FastAPI app for serving without Ray Serve."""
    local_app = FastAPI(
        title="Made With ML",
        description="Classify machine learning projects.",
        version="0.1",
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @local_app.get("/")
    def _index() -> Dict:
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @local_app.get("/run_id/")
    def _run_id() -> Dict:
        return {"run_id": run_id}

    @local_app.post("/evaluate/")
    async def _evaluate(request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @local_app.post("/predict/")
    async def _predict(request: Request, title: str = "", description: str = ""):
        data = await get_prediction_input(request=request, title=title, description=description)
        sample_ds = ray.data.from_items([{"title": data["title"], "description": data["description"], "tag": ""}])
        results = predict.predict_proba(ds=sample_ds, predictor=predictor)

        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < threshold:
                results[i]["prediction"] = "other"

        return {"results": make_json_safe(results)}

    return local_app


def make_json_safe(results):
    """Convert numpy values in prediction results to JSON-safe Python types."""
    safe_results = []
    for result in results:
        safe_results.append(
            {
                "prediction": result["prediction"],
                "probabilities": {label: float(prob) for label, prob in result["probabilities"].items()},
            }
        )
    return safe_results


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @app.post("/predict/")
    async def _predict(self, request: Request, title: str = "", description: str = ""):
        data = await get_prediction_input(request=request, title=title, description=description)
        sample_ds = ray.data.from_items([{"title": data["title"], "description": data["description"], "tag": ""}])
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return {"results": make_json_safe(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    parser.add_argument("--server", choices=["uvicorn", "ray"], default="uvicorn", help="server backend to use.")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="port to bind.")
    args = parser.parse_args()
    github_username = os.environ.get("GITHUB_USERNAME", "local")
    ray.init(num_gpus=0, runtime_env={"env_vars": {"GITHUB_USERNAME": github_username}})
    if args.server == "ray":
        serve.start(http_options={"host": args.host, "port": args.port})
        serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))
    else:
        uvicorn.run(create_app(run_id=args.run_id, threshold=args.threshold), host=args.host, port=args.port)
