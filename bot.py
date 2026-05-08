import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
import time

# -----------------------------
# TELEGRAM SETTINGS
# -----------------------------
TOKEN = "8043643820:AAFOd4Aa1nfOjwekOpTCsCOhJN-xAXuMN3w"
CHAT_ID = "8250122936"

bot = Bot(token=TOKEN)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------
def get_data(symbol="AAPL"):

    df = yf.download(
        tickers=symbol,
        period="1mo",
        interval="1h",
        progress=False
    )

    if df is None or df.empty:
        return None

    df = df.reset_index()

    if "Close" not in df.columns:
        return None

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df.dropna(inplace=True)

    if len(df) < 20:
        return None

    close_series = pd.Series(df["Close"].values.flatten())

    df["rsi"] = RSIIndicator(close_series).rsi()

    df["future"] = df["Close"].shift(-3)

    df["target"] = np.where(
        df["future"] > df["Close"],
        1,
        0
    )

    df.dropna(inplace=True)

    return df

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(df):

    X = df[["Close", "Volume", "rsi"]]
    y = df["target"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model

# -----------------------------
# GENERATE SIGNAL
# -----------------------------
def generate_signal(model, df):

    latest = pd.DataFrame([{
        "Close": df.iloc[-1]["Close"],
        "Volume": df.iloc[-1]["Volume"],
        "rsi": df.iloc[-1]["rsi"]
    }])

    prediction = model.predict(latest)[0]

    probability = model.predict_proba(latest)[0][prediction]

    if probability < 0.75:
        return "WAIT", probability

    if prediction == 1:
        return "BUY", probability

    return "SELL", probability

# -----------------------------
# SEND TELEGRAM MESSAGE
# -----------------------------
def send_signal(symbol, signal, confidence):

    text = f"""
📊 AI TRADING SIGNAL

📈 Symbol: {symbol}
📌 Signal: {signal}
🧠 Confidence: {round(confidence * 100, 2)}%
"""

    bot.send_message(
        chat_id=CHAT_ID,
        text=text
    )

# -----------------------------
# MAIN LOOP
# -----------------------------
SYMBOL = "AAPL"

while True:

    try:

        print("Downloading market data...")

        df = get_data(SYMBOL)

        if df is None:
            print("No valid market data")
            time.sleep(60)
            continue

        print("Training AI model...")

        model = train_model(df)

        signal, confidence = generate_signal(model, df)

        print(signal, confidence)

        if signal != "WAIT":
            send_signal(
                SYMBOL,
                signal,
                confidence
            )

    except Exception as e:
        print("ERROR:", e)

    time.sleep(600)
