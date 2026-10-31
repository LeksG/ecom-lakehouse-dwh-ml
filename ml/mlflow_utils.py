"""Спільні функції MLflow: трекінг експериментів та реєстр моделей."""
import mlflow
from mlflow import MlflowClient

from config import MLFLOW_URI, MLFLOW_ARTIFACTS


def setup(experiment: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACTS.as_uri() + "/" + experiment)
    mlflow.set_experiment(experiment)


def promote(model_name: str, version: str, alias: str = "champion"):
    """Призначення аліасу (champion) зареєстрованій версії моделі."""
    MlflowClient().set_registered_model_alias(model_name, alias, version)


# Моделі створюються в межах проєкту, тому внутрішні типи дерев sklearn позначаються як довірені для skops
TRUSTED = ["sklearn.tree._tree.Tree", "sklearn._loss.link.Interval", "sklearn._loss.link.IdentityLink",
           "sklearn._loss.link.LogitLink", "sklearn._loss.loss.HalfSquaredError",
           "sklearn._loss.loss.HalfBinomialLoss", "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
           "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "numpy.dtype"]


def log_sklearn(model, X, y_pred, name="model"):
    from mlflow.models import infer_signature
    return mlflow.sklearn.log_model(model, name=name, signature=infer_signature(X, y_pred),
                                    input_example=X.head(3), skops_trusted_types=TRUSTED)


def load_sklearn(uri):
    return mlflow.sklearn.load_model(uri)
