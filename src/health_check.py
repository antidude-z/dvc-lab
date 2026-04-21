import json

import numpy as np
import pandas as pd
import requests

SAMPLE_COUNT = 50


def main():
    X_test = pd.read_csv("data/X_test_fe.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    np.random.seed(42)
    sample_idx = np.random.choice(X_test.index, size=SAMPLE_COUNT, replace=False)

    X_sample = X_test.loc[sample_idx].copy()
    y_sample = y_test.loc[sample_idx]

    print("\nОтправляем запрос в FastAPI модель...")

    correct = 0

    for i, (idx, row) in enumerate(X_sample.iterrows()):
        data_payload = {"input": row.tolist()}

        response = requests.post(
            "http://localhost:8080/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data_payload),
        )
        response.raise_for_status()

        pred = response.json()["prediction"][0]

        true_ans = y_sample.loc[idx]

        if true_ans == pred:
            correct += 1

        print(f"Пример {i + 1:2d} | Истинный: {true_ans:10} | Предсказание: {pred}")

    print(f"\nВерных предсказаний: {correct * 100 / SAMPLE_COUNT:.2f}%")


if __name__ == "__main__":
    main()
