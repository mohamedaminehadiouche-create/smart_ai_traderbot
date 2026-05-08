import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
import time

# -------------------------
# TELEGRAM CONFIG
# -------------------------
TOKEN = "8043643820:AAFOd4Aa1nfOjwekOpTCsCOhJN-xAXuMN3w"
CHAT_ID = "8250122936"

bot = Bot(token=TOKEN)

# -------------------------
# MARKET DATA
# -------------------------
def get_data(symbol="AAPL"):
    df = yf.download(symbol, period="6mo", interval="1h")
    df.dropna(inplace=True)

    df["rsi"] = RSIIndicator(df["Close"]).rsi()

    df["future"] = df["Close"].shift(-3)
    df["target"] = np.where(df["future"] > df["Close"], 1, 0)

    df.dropna(inplace=True)
    return df

# -------------------------
# AI MODEL
# -------------------------
def train_model(df):
    X = df[["Close", "Volume", "rsi"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)

    return model

# -------------------------
# SIGNAL GENERATION
# -------------------------
def generate_signal(model, df):
    latest = df.iloc[-1][["Close", "Volume", "rsi"]].values.reshape(1, -1)

    prediction = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][prediction]

    if prediction == 1 and prob > 0.75:
        return "BUY", prob

    elif prediction == 0 and prob > 0.75:
        return "SELL", prob

    else:
        return "WAIT", prob

# -------------------------
# SEND MESSAGE
# -------------------------
def send_message(symbol, signal, confidence):
    message = f"""
📊 AI TRADING SIGNAL

Symbol: {symbol}
Signal: {signal}
Confidence: {round(confidence*100, 2)}%
"""

    bot.send_message(chat_id=CHAT_ID, text=message)

# -------------------------
# MAIN LOOP
# -------------------------
symbol = "AAPL"

df = get_data(symbol)
model = train_model(df)

while True:
    df = get_data(symbol)

    signal, conf = generate_signal(model, df)

    if signal != "WAIT":
        send_message(symbol, signal, conf)

    time.sleep(600)
