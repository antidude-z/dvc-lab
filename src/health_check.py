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

    data_payload = {"inputs": X_sample.to_dict(orient="records")}

    print("\nОтправляем запрос в MLflow модель...")

    response = requests.post(
        "http://localhost:8080/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data_payload),
    )
    response.raise_for_status()
    preds = response.json()["predictions"]

    print("\nРезультаты предсказаний:")
    print("-" * 50)

    correct = 0
    for i, (idx, _row) in enumerate(X_sample.iterrows()):
        true_ans = y_sample.loc[idx]
        pred = preds[i]

        if true_ans == pred:
            correct += 1

        print(
            f"Пример {i + 1:2d} | "
            f"Истинный: {y_sample.loc[idx]:10} | "
            f"Предсказание: {preds[i]}"
        )

    print(f"Верных предсказаний: {correct * 100 / SAMPLE_COUNT}%")


if __name__ == "__main__":
    main()
