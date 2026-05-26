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
    # 1. MENANGKAP PARAMETER
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    print(f"🚀 Menjalankan Re-training: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # 2. INISIALISASI
    print("🌐 Menyambungkan ke remote tracker DagsHub...")
    dagshub.init(repo_owner="ariefwcks303", repo_name="Eksperimen_SML_Arief", mlflow=True)
    mlflow.set_experiment("Student_Placement_Advance")
    
    df = pd.read_csv("student_dataset_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. PELATIHAN
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 4. LOGGING & PENYIMPANAN LOKAL
    with mlflow.start_run(run_name="CI_Automated_Retraining") as run:
        print("📝 Mencatat data ke MLflow & Menyimpan artefak lokal...")
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log ke DagsHub (Cloud)
        mlflow.sklearn.log_model(model, "model")
        
        # --- PERUBAHAN KRUSIAL ---
        # Simpan secara fisik di folder 'model' agar bisa dibaca Docker saat build
        # Jika folder 'model' sudah ada, hapus dulu agar tidak konflik
        if os.path.exists("model"):
            import shutil
            shutil.rmtree("model")
        mlflow.sklearn.save_model(model, "model")
        # -------------------------
        
        # Plotting
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Simpan ID untuk referensi
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print("🎯 Selesai! Folder 'model' telah siap untuk dibungkus Docker.")

if __name__ == "__main__":
    main()