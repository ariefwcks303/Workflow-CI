FROM python:3.10-slim
WORKDIR /app
# Copy model yang sudah didownload ke folder /app/model
COPY model /app/model
RUN pip install mlflow scikit-learn pandas
EXPOSE 8080
# Perintah untuk menjalankan model dengan MLflow server
CMD ["mlflow", "models", "serve", "-m", "model", "--host", "0.0.0.0", "--port", "8080"]