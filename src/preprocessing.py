import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def main():
    # 1. Загрузка данных
    dataset = fetch_ucirepo(id=697)
    df = dataset.data.original

    # 2. Целевая переменная
    y = (df["Target"] == "Dropout").astype(int)
    X = df.drop(columns=["Target"])

    # 3. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Сохранение в CSV для DVC
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["Target"]).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["Target"]).to_csv("data/y_test.csv", index=False)


if __name__ == "__main__":
    main()
