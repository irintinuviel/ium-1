import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
RESULTS_PATH = os.getenv("RESULTS_PATH", "wyniki.txt")


def load_data(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        df = load_iris(as_frame=True).frame
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


def main() -> None:
    df = load_data(DATA_PATH)
    print("Pierwsze wiersze:\n", df.head(), "\n")
    print("Rozmiar zbioru:", df.shape, "\n")
    print("Statystyki opisowe:\n", df.describe(), "\n")

    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Zbiór treningowy: {X_train.shape}, testowy: {X_test.shape}\n")

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=load_iris().target_names
    )
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Wyniki zapisano do {RESULTS_PATH}")


if __name__ == "__main__":
    main()
