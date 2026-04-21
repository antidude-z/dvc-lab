import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Загрузка промежуточных данных
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")["Target"]
    y_test = pd.read_csv("data/y_test.csv")["Target"]

    # 2. Генерация новых признаков (2-3 штуки)
    # Признак 1: Динамика оценок между семестрами
    c1, c2 = "Curricular_units_1st_sem_(grade)", "Curricular_units_2nd_sem_(grade)"
    if c1 in X_train.columns and c2 in X_train.columns:
        X_train["Grade_Trend"] = X_train[c2] - X_train[c1]
        X_test["Grade_Trend"] = X_test[c2] - X_test[c1]

    # Признак 2: Эффективность сдачи (сданные / попытки)
    c_app, c_eval = "Curricular_units_1st_sem_(approved)", "Curricular_units_1st_sem_(evaluations)"
    if c_app in X_train.columns and c_eval in X_train.columns:
        X_train["Pass_Rate_Sem1"] = X_train[c_app] / (X_train[c_eval] + 1e-6)
        X_test["Pass_Rate_Sem1"] = X_test[c_app] / (X_test[c_eval] + 1e-6)

    # Признак 3: Административно-финансовый риск
    c_deb, c_fee = "Debtor", "Tuition fees up to date"
    if c_deb in X_train.columns and c_fee in X_train.columns:
        X_train["Admin_Risk_Score"] = X_train[c_deb] + (1 - X_train[c_fee])
        X_test["Admin_Risk_Score"] = X_test[c_deb] + (1 - X_test[c_fee])

    # 3. Разделение на категории и числа
    categorical_cols = [
        "Marital Status", "Application mode", "Application order", "Course",
        "Daytime/evening attendance", "Previous qualification", "Nacionality",
        "Mother's qualification", "Father's qualification", "Mother's occupation",
        "Father's occupation", "Displaced", "Educational special needs",
        "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
    ]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # 4. One-Hot Encoding
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 5. Масштабирование числовых признаков
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 6. Сохранение финальных данных
    X_train.to_csv("data/X_train_fe.csv", index=False)
    X_test.to_csv("data/X_test_fe.csv", index=False)
    pd.DataFrame(y_train, columns=["Target"]).to_csv("data/y_train_fe.csv", index=False)
    pd.DataFrame(y_test, columns=["Target"]).to_csv("data/y_test_fe.csv", index=False)

if __name__ == "__main__":
    main()