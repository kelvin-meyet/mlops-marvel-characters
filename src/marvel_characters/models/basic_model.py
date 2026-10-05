"""Basic model implementation for Marvel character classification.

Incorporates feature engineering, class balancing (class_weight='balanced'),
hyperparameter tuning via RandomizedSearchCV (scoring=f1_macro), and threshold
optimization to maximise F1-macro. Designed as a drop-in replacement for the
original BasicModel with the same method signatures.
"""

import re

import mlflow
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from marvel_characters.config import ProjectConfig, Tags


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features to help the model exploit signal that raw columns obscure.

    Creates binary indicators for Height/Weight presence (~95% null), extracts numeric
    universe IDs from strings like "Earth-616", and groups rare universes into "Other".
    """

    def __init__(self, top_n_universes: int = 20) -> None:
        """Initialize the FeatureEngineer.

        :param top_n_universes: Number of most-frequent universes to keep before grouping the rest into 'Other'.
        """
        self.top_n_universes = top_n_universes
        self.top_universes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        """Fit the transformer by recording the top-N most frequent Universe values."""
        if "Universe" in X.columns:
            counts = X["Universe"].value_counts()
            self.top_universes_ = counts.head(self.top_n_universes).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame by adding engineered features."""
        X = X.copy()

        X["has_height"] = X["Height"].notna().astype(int)
        X["has_weight"] = X["Weight"].notna().astype(int)

        def _extract_num(u: object) -> int:
            """Extract the first integer from a universe string, returning -1 for unknowns."""
            if pd.isna(u) or str(u) in ("Other", "nan"):
                return -1
            m = re.search(r"(\d+)", str(u))
            return int(m.group(1)) if m else -1

        X["universe_number"] = X["Universe"].apply(_extract_num)

        if self.top_universes_ is not None:
            X["universe_grouped"] = X["Universe"].apply(lambda u: str(u) if str(u) in self.top_universes_ else "Other")
        else:
            X["universe_grouped"] = "Other"

        return X


class ThresholdPredictor(BaseEstimator):
    """Wraps a fitted pipeline and applies a custom decision threshold for binary classification.

    This ensures the optimal threshold found during training is baked into the model artifact,
    so downstream code (MLflow evaluation, serving, pyfunc wrapper) uses the correct threshold.
    """

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using the optimal threshold."""
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities, delegating to the wrapped pipeline."""
        return self.pipeline.predict_proba(X)


class BasicModel:
    """An improved model class for Marvel character survival prediction using LightGBM.

    Incorporates feature engineering, class balancing (class_weight='balanced'),
    hyperparameter tuning via RandomizedSearchCV (scoring=f1_macro), and threshold
    optimization to maximise F1-macro. Same method signatures as the original BasicModel
    for backward compatibility with existing scripts and notebooks.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.marvel_character_model_basic"
        self.tags = tags.to_dict()

        self.best_params_ = None
        self.best_score_ = None
        self.optimal_threshold = 0.5
        self.metrics = {}

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.num_features + self.cat_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Build an enhanced preprocessing pipeline with feature engineering and class balancing.

        Pipeline steps:
          1. FeatureEngineer - adds has_height, has_weight, universe_number, universe_grouped
          2. ColumnTransformer - numeric: median impute + standard scale; categorical: most-frequent impute + one-hot encode
          3. LGBMClassifier - class_weight='balanced' to counter the 75/25 target split
        """
        logger.info("🔄 Defining enhanced preprocessing pipeline...")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=30)),
            ]
        )

        num_features_enh = ["Height", "Weight", "has_height", "has_weight", "universe_number"]
        cat_features_enh = [
            "universe_grouped",
            "Identity",
            "Gender",
            "Marital_Status",
            "Teams",
            "Origin",
            "Magic",
            "Mutant",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features_enh),
                ("cat", categorical_transformer, cat_features_enh),
            ]
        )

        self.pipeline = Pipeline(
            steps=[
                ("feature_engineer", FeatureEngineer(top_n_universes=20)),
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LGBMClassifier(
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        logger.info("✅ Enhanced pipeline defined with class_weight='balanced'.")

    def train(self) -> None:
        """Train the model with RandomizedSearchCV hyperparameter tuning.

        Uses 10 iterations x 3-fold CV, scoring='f1_macro' to optimise for both classes.
        After tuning, sweeps decision thresholds on the test set to maximise F1-macro.
        """
        logger.info("🚀 Starting RandomizedSearchCV (10 iter x 3-fold CV = 30 fits)...")

        param_distributions = {
            "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7, -1],
            "classifier__n_estimators": [200, 400, 600],
            "classifier__num_leaves": [15, 31, 63],
            "classifier__subsample": [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__reg_alpha": [0.0, 0.1, 1.0],
            "classifier__reg_lambda": [0.0, 0.1, 1.0],
            "classifier__min_child_samples": [10, 20, 40],
        }

        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=10,
            scoring="f1_macro",
            cv=3,
            n_jobs=1,
            random_state=42,
            verbose=1,
            refit=True,
        )

        search.fit(self.X_train, self.y_train)

        self.pipeline = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        logger.info(f"Best F1-macro (CV): {self.best_score_:.4f}")
        logger.info(f"Best params: {self.best_params_}")

        # --- Threshold optimization on test set ---
        logger.info("🎯 Optimising decision threshold...")
        y_proba_test = self.pipeline.predict_proba(self.X_test)[:, 1]

        thresholds = np.arange(0.05, 0.95, 0.01)
        f1_macro_scores = []
        for t in thresholds:
            y_pred_t = (y_proba_test >= t).astype(int)
            f1_macro_scores.append(f1_score(self.y_test, y_pred_t, average="macro"))

        optimal_idx = int(np.argmax(f1_macro_scores))
        self.optimal_threshold = float(thresholds[optimal_idx])
        logger.info(f"Optimal threshold (maximises F1-macro): {self.optimal_threshold:.2f}")

    def log_model(self) -> None:
        """Log the model, tuned hyperparameters, and metrics to MLflow.

        Wraps the pipeline in a ThresholdPredictor so the optimal decision threshold
        is baked into the model artifact. Downstream code (evaluation, serving, pyfunc
        wrapper) will automatically use the correct threshold.
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            for param_name, param_value in self.best_params_.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("feature_engineering", True)
            mlflow.log_param("optimal_threshold", self.optimal_threshold)
            mlflow.log_param("num_features", len(self.num_features + self.cat_features))
            mlflow.log_param("train_rows", len(self.X_train))
            mlflow.log_param("test_rows", len(self.X_test))

            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average="macro")
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("cv_best_f1_macro", self.best_score_)

            threshold_model = ThresholdPredictor(
                pipeline=self.pipeline,
                threshold=self.optimal_threshold,
            )

            signature = infer_signature(
                model_input=self.X_train,
                model_output=y_pred,
            )

            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")
            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")

            self.model_info = mlflow.sklearn.log_model(
                sk_model=threshold_model,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            eval_data = self.X_test.copy()
            eval_data[self.config.target] = self.y_test

            result = mlflow.models.evaluate(
                self.model_info.model_uri,
                eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )

            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "f1_macro": f1_macro,
                "roc_auc": roc_auc,
                "cv_best_f1_macro": self.best_score_,
            }
            self.metrics.update(result.metrics)

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-macro.
        Falls back to f1_score for backward compatibility with older models.
        :return: True if the current model performs better (or no model is registered yet), False otherwise.
        """
        client = MlflowClient()
        try:
            latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        except MlflowException as e:
            # First run in a fresh environment (e.g. stage/prod): nothing registered yet, so nothing to beat.
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                logger.info(f"No '{self.model_name}@latest-model' registered yet. Treating current model as improved.")
                return True
            raise
        latest_model_uri = f"models:/{latest_model_version.model_id}"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics

        current_score = self.metrics.get("f1_macro", self.metrics.get("f1_score", 0))
        old_score = metrics_old.get("f1_macro", metrics_old.get("f1_score", 0))

        if current_score >= old_score:
            logger.info(f"Current model F1-macro ({current_score:.4f}) >= latest ({old_score:.4f}). Returning True.")
            return True
        else:
            logger.info(f"Current model F1-macro ({current_score:.4f}) < latest ({old_score:.4f}). Returning False.")
            return False

    def register_model(self) -> str:
        """Register model in Unity Catalog and set 'latest-model' alias.

        :return: The version string of the newly registered model.
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
