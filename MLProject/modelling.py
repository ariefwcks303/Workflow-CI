import os
import argparse
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():

    # =========================
    # PARAMETER
    # =========================
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)

    args = parser.parse_args()

    print(
        f"🚀 Training: n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth}"
    )

    # =========================
    # MLFLOW CONFIG
    # =========================
    print("🌐 Connecting to DagsHub MLflow Tracking Server...")

    mlflow.set_tracking_uri(
        "https://dagshub.com/ariefwcks303/Eksperimen_SML_Arief.mlflow"
    )

    mlflow.set_experiment("Student_Placement_Advanced_Online")

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv("student_dataset_clean.csv")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================
    # TRAIN MODEL
    # =========================
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"✅ Accuracy: {accuracy:.4f}")

    # =========================
    # START MLFLOW RUN
    # =========================
    with mlflow.start_run(run_name="CI_Automated_Retraining") as run:

        # PARAMETERS
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # METRICS
        mlflow.log_metric("accuracy", accuracy)

        # =========================
        # CONFUSION MATRIX
        # =========================
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")

        plt.savefig("confusion_matrix.png")

        mlflow.log_artifact("confusion_matrix.png")

        # =========================
        # LOG MODEL TO DAGSHub
        # =========================
        mlflow.sklearn.log_model(model, "model")

        # =========================
        # SAVE MODEL LOCAL
        # =========================
        if os.path.exists("model"):
            shutil.rmtree("model")

        mlflow.sklearn.save_model(model, "model")

        # =========================
        # SAVE RUN ID
        # =========================
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print("✅ Model saved locally.")
        print("✅ Model logged to DagsHub.")
        print(f"🆔 Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
