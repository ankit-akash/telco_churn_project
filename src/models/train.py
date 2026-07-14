import os
import shutil
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# ===========================================================
# MLflow Configuration
# ===========================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/mlflow.db")

ARTIFACT_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def train_model(df: pd.DataFrame, target_col: str):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        # ======================================
        # Save feature columns
        # ======================================

        feature_file = os.path.join(
            ARTIFACT_DIR,
            "feature_columns.txt"
        )

        with open(feature_file, "w") as f:
            for col in X.columns:
                f.write(col + "\n")

        mlflow.log_artifact(feature_file)

        # ======================================
        # Save preprocessing object
        # ======================================

        preprocessing_file = os.path.join(
            ARTIFACT_DIR,
            "preprocessing.pkl"
        )

        joblib.dump({}, preprocessing_file)

        mlflow.log_artifact(preprocessing_file)

        print(f"\nAccuracy : {acc:.4f}")
        print(f"Recall   : {rec:.4f}")

        print("\nRun ID:")
        print(run.info.run_id)

        print("\nArtifacts saved.")

        # ======================================
        # Copy model to src/serving/model
        # ======================================

        run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()

        temp_download = os.path.join(ROOT_DIR, "_temp_mlflow")

        if os.path.exists(temp_download):
            shutil.rmtree(temp_download)

        os.makedirs(temp_download, exist_ok=True)

        client.download_artifacts(
            run_id=run_id,
            path="",
            dst_path=temp_download,
        )

        serving_dir = os.path.join(
            ROOT_DIR,
            "src",
            "serving",
            "model",
            run_id
        )

        if os.path.exists(serving_dir):
            shutil.rmtree(serving_dir)

        shutil.copytree(
            temp_download,
            serving_dir
        )

        shutil.rmtree(temp_download)

        print("\n✅ Model copied to:")
        print(serving_dir)