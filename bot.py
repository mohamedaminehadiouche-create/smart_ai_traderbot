import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
import time

# --------------------------------
# TELEGRAM CONFIG
# --------------------------------
TOKEN = "8043643820:AAFOd4Aa1nfOjwekOpTCsCOhJN-xAXuMN3w"
CHAT_ID = "8250122936"

bot = Bot(token=TOKEN)

# --------------------------------
# GET MARKET DATA
# --------------------------------
def get_data(symbol="AAPL"):
    df = yf.download(symbol, period="6mo", interval="1h")

    if df.empty:
        return None

    df.dropna(inplace=True)

    close_prices = df["Close"].squeeze()

    df["rsi"] = RSIIndicator(close_prices).rsi()

    df["future"] = df["Close"].shift(-3)

    df["target"] = np.where(df["future"] > df["Close"], 1, 0)

    df.dropna(inplace=True)

    return df

# --------------------------------
# TRAIN AI MODEL
# --------------------------------
def train_model(df):
    X = df[["Close", "Volume", "rsi"]]
    y = df["target"]

    model = RandomForestClassifier(n_estimators=150)

    model.fit(X, y)

    return model

# --------------------------------
# GENERATE SIGNAL
# --------------------------------
def generate_signal(model, df):

    latest = np.array([
        df.iloc[-1]["Close"],
        df.iloc[-1]["Volume"],
        df.iloc[-1]["rsi"]
    ]).reshape(1, -1)

    prediction = model.predict(latest)[0]

    prob = model.predict_proba(latest)[0][prediction]

    if prediction == 1 and prob > 0.75:
        return "BUY", prob

    elif prediction == 0 and prob > 0.75:
        return "SELL", prob

    else:
        return "WAIT", prob

# --------------------------------
# SEND TELEGRAM MESSAGE
# --------------------------------
def send_message(symbol, signal, confidence):

    message = f'''
📊 AI TRADING SIGNAL

Symbol: {symbol}

Signal: {signal}

Confidence: {round(confidence * 100, 2)}%
'''

    bot.send_message(
        chat_id=CHAT_ID,
        text=message
    )

# --------------------------------
# MAIN LOOP
# --------------------------------
symbol = "AAPL"

while True:

    try:

        df = get_data(symbol)

        if df is None:
            print("No data received")
            time.sleep(60)
            continue

        model = train_model(df)

        signal, conf = generate_signal(model, df)

        print(signal, conf)

        if signal != "WAIT":
            send_message(symbol, signal, conf)

    except Exception as e:
        print("ERROR:", e)

    time.sleep(600)
