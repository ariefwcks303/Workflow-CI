import os
import argparse
import pandas as pd
import mlflow
import dagshub
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def main():
    # 1. TANGKAP PARAMETER
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    print(f"🚀 Training: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # 2. INISIALISASI (FORCE ENV VARIABLE)
    # Ini kuncinya: Kita bypass interaktif dengan memastikan token ada di environment
    # dagshub.init akan otomatis mendeteksi DAGSHUB_USER_TOKEN dari env
    print("🌐 Menyambungkan ke DagsHub via Token...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    mlflow.set_experiment("Student_Placement_Advance")
    
    # 3. DATA & TRAINING
    df = pd.read_csv("student_dataset_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. LOGGING & PENYIMPANAN
    with mlflow.start_run(run_name="CI_Automated_Retraining") as run:
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
        
        # Log ke DagsHub
        mlflow.sklearn.log_model(model, "model")
        
        # Simpan LOKAL untuk Docker
        if os.path.exists("model"):
            shutil.rmtree("model")
        mlflow.sklearn.save_model(model, "model")
        
        # Simpan ID
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print("✅ Selesai! Model tersimpan lokal dan di-log ke DagsHub.")

if __name__ == "__main__":
    main()