FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir scikit-learn fastapi uvicorn

COPY data/model.pkl .
COPY docker/app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
