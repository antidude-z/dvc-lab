FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow/best_model/model /model

EXPOSE 8080

CMD ["mlflow", "models", "serve", "-m", "/model", "-p", "8080", "--host", "0.0.0.0", "--env-manager=local"]
