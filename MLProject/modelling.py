import os
import argparse
import pandas as pd
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def main():
    # -------------------------------------------------------------------------
    # 1. MENANGKAP PARAMETER DARI MLPROJECT (ARGPARSE)
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    print(f"🚀 Menjalankan Re-training dengan Parameter -> n_estimators: {args.n_estimators}, max_depth: {args.max_depth}")

    # -------------------------------------------------------------------------
    # 2. INISIALISASI SAMBUNGAN DAGSHUB (ONLINE)
    # -------------------------------------------------------------------------
    print("🌐 Menyambungkan ke remote tracker DagsHub...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    mlflow.set_experiment("Student_Placement_Advance")
    
    # Membaca dataset yang berada di folder yang sama
    df = pd.read_csv("student_dataset_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # -------------------------------------------------------------------------
    # 3. PELATIHAN MODEL (MENGGUNAKAN VALUE DARI ARGUMEN)
    # -------------------------------------------------------------------------
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # -------------------------------------------------------------------------
    # 4. LOGGING HASIL OTOMATISASI KE MLFLOW
    # -------------------------------------------------------------------------
    with mlflow.start_run(run_name="CI_Automated_Retraining") as run:
        print("📝 Mencatat parameter, metrik, dan artefak baru...")
        
        # Mencatat parameter yang digunakan saat ini
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        # Mencatat metrik performa
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Membuat dan mengunggah Artefak 1: Model biner (.pkl)
        model_file = "best_model.pkl"
        joblib.dump(model, model_file)
        mlflow.log_artifact(model_file)
        
        # Membuat dan mengunggah Artefak 2: Plot matriks evaluasi (.png)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
        plt.title('Confusion Matrix CI Automated Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plot_file = "confusion_matrix.png"
        plt.savefig(plot_file)
        plt.close()
        
        mlflow.log_artifact(plot_file)
        
        # 🌟 LOG TAKTIS: Tulis RUN_ID aktif ke dalam file teks sebelum session ditutup
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print("🎯 Selesai! Semua data sukses terkirim ke cloud DagsHub.")

if __name__ == "__main__":
    main()