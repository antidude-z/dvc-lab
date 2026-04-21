#!/bin/bash
set -e

docker rm -f mlflow-model 2>/dev/null || true

docker run -d -p 8080:8080 --name mlflow-model student-dropout-model

sleep 10