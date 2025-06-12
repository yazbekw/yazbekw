import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# تحسين استخدام ذاكرة GPU إن وجدت
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import schedule
import time
from flask import Flask, render_template, jsonify
import threading
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import logging

# إعدادات التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# === متغيرات للتخزين المؤقت ===
cached_market_data = {
    'balance': None,
    'ticker': None,
    'ohlcv': None,
    'last_prediction': None,
    'last_updated': None
}

# === وظائف التداول ===
def get_live_data():
    """جلب بيانات السوق الحية"""
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"فشل جلب البيانات: {e}")
        # إرجاع بيانات مخزنة مؤقتاً كحل بديل
        return cached_market_data.get('ohlcv', pd.DataFrame())

# تجهيز Scaler واحد فقط
scaler = MinMaxScaler()
features = ['open', 'high', 'low', 'close', 'volume', 'MA_10', 'MA_50', 'RSI']

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
    
    data = df[features].tail(24)
    
    if len(data) < 24:
        raise ValueError("لا يوجد بيانات كافية (24 ساعة مطلوبة)")
    
    # استخدام Scaler واحد مع التحقق من تركيبته
    if not hasattr(prepare_features, 'fitted'):
        scaler.fit(data)
        prepare_features.fitted = True
    
    scaled = scaler.transform(data)
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
            logging.info(f"[BUY] {amount:.8f} BTC at ${current_price:.2f}")
        except Exception as e:
            logging.error(f"[BUY ERROR] {e}")
    
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
            logging.info(f"[SELL] {open_trade['amount']:.8f} BTC at ${current_price:.2f}")
            logging.info(f"Profit: ${profit:.2f}")
            open_trade['is_open'] = False
        except Exception as e:
            logging.error(f"[SELL ERROR] {e}")

# === وظائف التحديث ===
def update_market_data():
    """تحديث بيانات السوق المخزنة مؤقتاً"""
    try:
        cached_market_data['balance'] = exchange.fetch_balance()
        cached_market_data['ticker'] = exchange.fetch_ticker(symbol)
        cached_market_data['ohlcv'] = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        cached_market_data['last_updated'] = datetime.now()
        logging.info("تم تحديث بيانات السوق بنجاح")
    except Exception as e:
        logging.error(f"فشل تحديث بيانات السوق: {e}")

def update_prediction():
    """تحديث الإشارة التنبؤية المخزنة مؤقتاً"""
    try:
        cached_market_data['last_prediction'] = predict_signal()
        logging.info(f"تم تحديث التنبؤ: {cached_market_data['last_prediction']}")
    except Exception as e:
        logging.error(f"فشل تحديث التنبؤ: {e}")

# === واجهة المستخدم ===
@app.route('/')
def dashboard():
    # استخدام البيانات المخزنة مؤقتاً
    balance = cached_market_data.get('balance', {})
    ticker = cached_market_data.get('ticker', {})
    
    # معالجة الرصيد
    processed_balance = {
        'free': {
            'USDT': balance.get('USDT', {}).get('free', 0) if balance else 0,
            'BTC': balance.get('BTC', {}).get('free', 0) if balance else 0
        },
        'total': balance.get('total', {}) if balance else {}
    }
    
    # حساب الربح الحالي إذا كانت هناك صفقة مفتوحة
    current_price = ticker.get('last', 0) if ticker else 0
    current_profit = 0
    if open_trade['is_open']:
        current_profit = (current_price - open_trade['buy_price']) * open_trade['amount']
    
    # الحصول على آخر إشارة
    last_signal = "غير معروف"
    if cached_market_data['last_prediction'] is not None:
        last_signal = "شراء" if cached_market_data['last_prediction'] == 1 else "بيع"
    
    # حساب وقت الفحص التالي
    next_check = (datetime.now() + timedelta(minutes=60 - datetime.now().minute)).strftime("%H:%M:%S")
    
    return render_template('dashboard.html',
                        open_trade=open_trade,
                        balance=processed_balance,
                        trades=trade_history,
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        last_signal=last_signal,
                        next_check=next_check,
                        last_updated=cached_market_data.get('last_updated', datetime.now()).strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/dashboard-data')
def dashboard_data():
    # استخدام البيانات المخزنة مؤقتاً
    balance = cached_market_data.get('balance', {})
    ticker = cached_market_data.get('ticker', {})
    
    processed_balance = {
        'free': {
            'USDT': balance.get('USDT', {}).get('free', 0) if balance else 0,
            'BTC': balance.get('BTC', {}).get('free', 0) if balance else 0
        },
        'total': balance.get('total', {}) if balance else {}
    }
    
    current_price = ticker.get('last', 0) if ticker else 0
    current_profit = 0
    if open_trade['is_open']:
        current_profit = (current_price - open_trade['buy_price']) * open_trade['amount']
    
    last_signal = "غير معروف"
    if cached_market_data['last_prediction'] is not None:
        last_signal = "شراء" if cached_market_data['last_prediction'] == 1 else "بيع"
    
    next_check = (datetime.now() + timedelta(minutes=60 - datetime.now().minute)).strftime("%H:%M:%S")
    
    return jsonify({
        'open_trade': open_trade,
        'balance': processed_balance,
        'trades': trade_history,
        'current_price': current_price,
        'current_profit': current_profit,
        'last_signal': last_signal,
        'next_check': next_check,
        'last_updated': cached_market_data.get('last_updated', datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    })

# === تشغيل البوت ===
def trading_job():
    """المهمة الدورية للتداول"""
    logging.info("جارٍ التحقق من السوق...")
    try:
        signal = predict_signal()
        execute_trade(signal)
    except Exception as e:
        logging.error(f"[ERROR] {e}")

def run_bot():
    """تشغيل البوت في الخلفية"""
    # التهيئة الأولية
    update_market_data()
    update_prediction()
    
    # الجدولة
    schedule.every().hour.at(":00").do(trading_job)  # التداول كل ساعة
    schedule.every(5).minutes.do(update_market_data)  # تحديث البيانات كل 5 دقائق
    schedule.every(30).minutes.do(update_prediction)  # تحديث التنبؤ كل 30 دقيقة
    
    logging.info("بدأ تشغيل بوت التداول")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    # بدء البوت في خيط منفصل
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # بدء خادم Flask
    logging.info("بدأ تشغيل خادم الويب")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), use_reloader=False)
