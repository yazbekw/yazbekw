import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import schedule
import time
from flask import Flask, render_template
import threading
from datetime import datetime

app = Flask(__name__)

# === إعدادات CoinEx API ===
exchange = ccxt.coinex({
    'apiKey': 'YOUR_API_KEY_HERE',
    'secret': 'YOUR_SECRET_KEY_HERE',
})

# === تحميل النموذج ===
model = load_model('yazbekw.keras')

# === إعدادات التداول ===
symbol = 'BTC/USDT'
investment_usdt = 9
open_trade = {
    'is_open': False,
    'buy_price': 0,
    'amount': 0,
    'buy_time': None
}
trade_history = []

# === الحصول على البيانات ===
def get_live_data():
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === تحضير البيانات ===
def prepare_features(df):
    features = df['close'].values[-50:]
    features = features.reshape(1, -1)
    return features

# === التنبؤ ===
def predict_signal():
    df = get_live_data()
    X = prepare_features(df)
    prediction = model.predict(X, verbose=0)
    return (prediction > 0.5).astype(int)[0][0]

# === تنفيذ الصفقة ===
def execute_trade(signal):
    global open_trade, trade_history
    
    current_price = exchange.fetch_ticker(symbol)['last']
    
    if signal == 1 and not open_trade['is_open']:  # شراء
        amount = investment_usdt / current_price
        try:
            order = exchange.create_market_buy_order(symbol, amount)
            open_trade = {
                'is_open': True,
                'buy_price': current_price,
                'amount': amount,
                'buy_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            trade_history.append({
                'type': 'BUY',
                'price': current_price,
                'amount': amount,
                'time': open_trade['buy_time']
            })
            print(f"[BUY] {amount:.8f} {symbol.split('/')[0]} at ${current_price:.2f}")
        except Exception as e:
            print(f"[BUY ERROR] {e}")
    
    elif signal == 0 and open_trade['is_open']:  # بيع
        try:
            order = exchange.create_market_sell_order(symbol, open_trade['amount'])
            profit = (current_price - open_trade['buy_price']) * open_trade['amount']
            trade_history.append({
                'type': 'SELL',
                'price': current_price,
                'amount': open_trade['amount'],
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'profit': profit
            })
            print(f"[SELL] Sold {open_trade['amount']:.8f} {symbol.split('/')[0]} at ${current_price:.2f}")
            print(f"Profit: ${profit:.2f}")
            open_trade['is_open'] = False
        except Exception as e:
            print(f"[SELL ERROR] {e}")

# === المهمة الدورية ===
def trading_job():
    print("جارٍ التحقق من السوق...")
    try:
        signal = predict_signal()
        execute_trade(signal)
    except Exception as e:
        print(f"[ERROR] حدث خطأ أثناء التشغيل: {e}")

# === واجهة المراقبة ===
@app.route('/')
def dashboard():
    balance = exchange.fetch_balance()
    return render_template('dashboard.html',
                         open_trade=open_trade,
                         balance=balance,
                         trades=trade_history,
                         symbol=symbol)

# === تشغيل البوت ===
def run_bot():
    schedule.every().hour.at(":00").do(trading_job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    # بدء البوت في خيط منفصل
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # بدء خادم Flask
    app.run(host='0.0.0.0', port=8000)
