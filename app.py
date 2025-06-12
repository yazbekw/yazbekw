import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import ccxt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import schedule
import time
from flask import Flask, render_template, jsonify  # تمت إضافة jsonify هنا
import threading
from datetime import datetime, timedelta  # تمت إضافة timedelta هنا
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
# ... (الاستيرادات الموجودة تبقى كما هي)

@app.route('/')
def dashboard():
    # إنشاء نسخة محلية من exchange للاستخدام في هذا الطلب
    local_exchange = ccxt.coinex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True
    })
    
    try:
        # جلب البيانات الحية
        balance = local_exchange.fetch_balance()
        ticker = local_exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # معالجة الرصيد
        processed_balance = {
            'free': {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'BTC': balance.get('BTC', {}).get('free', 0)
            },
            'total': balance.get('total', {})
        }
        
        # حساب الربح الحالي إذا كانت هناك صفقة مفتوحة
        current_profit = 0
        if open_trade['is_open']:
            current_profit = (current_price - open_trade['buy_price']) * open_trade['amount']
        
        # الحصول على آخر إشارة
        try:
            last_signal = "شراء" if predict_signal() else "بيع"
        except:
            last_signal = "غير معروف"
        
        # حساب وقت الفحص التالي (نفس منطق البوت)
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
                            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    except Exception as e:
        print(f"Error in dashboard: {e}")
        # في حالة الخطأ، نعود بيانات افتراضية
        return render_template('dashboard.html',
                            open_trade=open_trade,
                            balance={'free': {'USDT': 0, 'BTC': 0}, 'total': {}},
                            trades=trade_history,
                            symbol=symbol,
                            current_price=0,
                            current_profit=0,
                            last_signal="خطأ في الجلب",
                            next_check="--:--:--",
                            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/dashboard-data')
def dashboard_data():
    local_exchange = ccxt.coinex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True
    })
    
    try:
        balance = local_exchange.fetch_balance()
        ticker = local_exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        processed_balance = {
            'free': {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'BTC': balance.get('BTC', {}).get('free', 0)
            },
            'total': balance.get('total', {})
        }
        
        current_profit = 0
        if open_trade['is_open']:
            current_profit = (current_price - open_trade['buy_price']) * open_trade['amount']
        
        try:
            last_signal = "شراء" if predict_signal() else "بيع"
        except:
            last_signal = "غير معروف"
        
        next_check = (datetime.now() + timedelta(minutes=60 - datetime.now().minute)).strftime("%H:%M:%S")
        
        return jsonify({
            'open_trade': open_trade,
            'balance': processed_balance,
            'trades': trade_history,
            'current_price': current_price,
            'current_profit': current_profit,
            'last_signal': last_signal,
            'next_check': next_check,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        print(f"Error in dashboard data: {e}")
        return jsonify({
            'error': str(e),
            'open_trade': open_trade,
            'balance': {'free': {'USDT': 0, 'BTC': 0}, 'total': {}},
            'trades': trade_history,
            'current_price': 0,
            'current_profit': 0,
            'last_signal': "خطأ في الجلب",
            'next_check': "--:--:--",
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

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
