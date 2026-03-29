import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import os

LOG_FILE = "logs/pipeline.log"


def log(msg):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} | {msg}\n")


# Step 1: Load data
def load_data():
    df = pd.read_csv("data/prices.csv")
    return df


# Step 2: Feature engineering
def create_features(df):
    df['return'] = df['price'].pct_change()
    df['ma_5'] = df['price'].rolling(5).mean()
    df['ma_10'] = df['price'].rolling(10).mean()
    df['volatility'] = df['return'].rolling(5).std()

    df = df.dropna()
    return df


# Step 3: Create labels (simple signal)
def create_labels(df):
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    return df.dropna()


# Step 4: Train model
def train_model(df):
    features = ['ma_5', 'ma_10', 'volatility']
    X = df[features]
    y = df['target']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    log(f"Model trained. Accuracy: {acc:.4f}")

    return model


# Step 5: Backtest (very basic)
def backtest(df):
    df['strategy'] = df['target'] * df['return']
    pnl = df['strategy'].cumsum().iloc[-1]

    log(f"Backtest PnL: {pnl:.5f}")


def run():
    df = load_data()
    df = create_features(df)
    df = create_labels(df)

    train_model(df)
    backtest(df)


if __name__ == "__main__":
    run()
