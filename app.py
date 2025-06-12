import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # تعطيل تحذيرات TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # تقليل سجل TensorFlow

import tensorflow as tf
import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import schedule
import time
from flask import Flask, render_template
import threading
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()

# التحقق من وجود المفاتيح API
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("API keys missing! Please check your .env file")

app = Flask(__name__)

# تهيئة نموذج التداول
model = load_model("yazbekw.keras")

# إعدادات التداول
symbol = 'BTC/USDT'
investment_usdt = 9
open_trade = {
    'is_open': False,
    'buy_price': 0,
    'amount': 0,
    'buy_time': None
}
trade_history = []

# إنشاء اتصال بـ CoinEx
def create_exchange():
    return ccxt.coinex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'createMarketBuyOrderRequiresPrice': False}
    })

# اتصال رئيسي لعمليات التداول
exchange = create_exchange()

# === وظائف التداول ===
def get_live_data():
    """جلب بيانات السوق الحية"""
    bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def prepare_features(df):
    """تحضير البيانات للنموذج"""
    df = df.copy()
    
    # حساب المؤشرات الفنية
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()

    # حساب RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    
    features = ['open', 'high', 'low', 'close', 'volume', 'MA_10', 'MA_50', 'RSI']
    data = df[features].tail(24)
    
    if len(data) < 24:
        raise ValueError("لا يوجد بيانات كافية (24 ساعة مطلوبة)")
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return np.reshape(scaled, (1, 24, len(features)))

def predict_signal():
    """توليد إشارة التداول"""
    df = get_live_data()
    X = prepare_features(df)
    prediction = model.predict(X, verbose=0)
    return (prediction > 0.5).astype(int)[0][0]

def execute_trade(signal):
    """تنفيذ صفقة بناء على الإشارة"""
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
            print(f"[BUY] {amount:.8f} BTC at ${current_price:.2f}")
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
            print(f"[SELL] {open_trade['amount']:.8f} BTC at ${current_price:.2f}")
            print(f"Profit: ${profit:.2f}")
            open_trade['is_open'] = False
        except Exception as e:
            print(f"[SELL ERROR] {e}")

# === واجهة المستخدم ===
@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    local_exchange = create_exchange()
    
    try:
        balance = local_exchange.fetch_balance()
        processed_balance = {
            'free': {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'BTC': balance.get('BTC', {}).get('free', 0)
            },
            'total': balance.get('total', {})
        }
    except Exception as e:
        print(f"Error fetching balance: {e}")
        processed_balance = {'free': {'USDT': 0, 'BTC': 0}, 'total': {}}
    
    return render_template('dashboard.html',
                         open_trade=open_trade,
                         balance=processed_balance,
                         trades=trade_history,
                         symbol=symbol)

# === تشغيل البوت ===
def trading_job():
    """المهمة الدورية للتداول"""
    print("جارٍ التحقق من السوق...")
    try:
        signal = predict_signal()
        execute_trade(signal)
    except Exception as e:
        print(f"[ERROR] {e}")

def run_bot():
    """تشغيل البوت في الخلفية"""
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
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
