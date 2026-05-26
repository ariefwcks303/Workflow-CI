# 🚀 Student Placement Prediction Pipeline (MLOps)

Automated end-to-end Machine Learning pipeline using **MLflow, DagsHub, and Docker**.

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **ML Ops:** MLflow, DagsHub, GitHub Actions (CI/CD)
- **Containerization:** Docker, Docker Hub
- **Versioning:** Git

## ⚙️ How It Works

1. **CI/CD:** Automated training triggered on every `git push`.
2. **Tracking:** Experiment metrics and parameters logged remotely to **DagsHub**.
3. **Artifacts:** Local artifact saving (`save_model`) ensuring seamless Docker image build.
4. **Deployment:** Docker image automatically pushed to Docker Hub upon successful build.

## 🔗 Project Links

- **Experiment Tracking:** [DagsHub Dashboard](https://dagshub.com/ariefwcks303/Eksperimen_SML_Arief.mlflow)
- **Container Image:** [Docker Hub Repository](https://hub.docker.com/r/ariefwcksdevs/student-placement-model)

## 🏗 Project Structure

```text
.
├── .github/workflows/   # CI/CD Workflow
├── MLProject/           # Training scripts & Dockerfile
├── conda.yaml           # Dependency management
└── README.md
```
