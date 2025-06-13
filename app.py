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


# تجهيز Scaler واحد فقط
scaler = MinMaxScaler()
features = ['close', 'MA_10', 'MA_50', 'RSI', 'price_change', 'volatility', 'volume_change', 'volume']

def prepare_features(df):
    """إصدار محسن مع معالجة أفضل للبيانات الناقصة"""
    df = df.copy()
    
    # 1. حساب المؤشرات الفنية مع تعبئة القيم الناقصة
    df['MA_10'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['MA_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # 2. حساب RSI بشكل أكثر قوة
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).ffill().bfill()))
    
    # 3. ميزات إضافية
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['close'].rolling(window=24, min_periods=1).std().fillna(0)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    
    # 4. التأكد من عدم وجود قيم ناقصة
    df = df.dropna().tail(24)
    
    if len(df) < 24:
        raise ValueError(f"بيانات غير كافية بعد التنظيف ({len(df)}/24)")
    
    # 5. اختيار الميزات المطلوبة
    features = ['close', 'MA_10', 'MA_50', 'RSI', 'price_change', 'volatility', 'volume_change', 'volume']
    data = df[features]
    
    # 6. التحجيم
    if not hasattr(prepare_features, 'fitted'):
        scaler.fit(data)
        prepare_features.fitted = True
    
    scaled = scaler.transform(data)
    return np.reshape(scaled, (1, 24, 8))

    
def predict_direction():
    """توليد اتجاه التداول مع تجنب التكرار"""
    global cached_market_data
    df = get_live_data()
    X = prepare_features(df)
    predictions = model.predict(X, verbose=0)[0]
    direction = int(np.sign(predictions[-1] - predictions[0]))

    # تجاهل الإشارة إذا كانت مطابقة للإشارة السابقة
    if 'last_prediction' in cached_market_data:
        if direction == cached_market_data['last_prediction']:
            return 0
    return direction

def clear_unfilled_orders():
    """إلغاء جميع الأوامر غير المنفذة"""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logging.info("لا توجد أوامر غير منفذة")
            return

        for order in open_orders:
            exchange.cancel_order(order['id'], symbol)
            logging.info(f"[CANCEL] تم إلغاء الأمر: {order['id']} - {order['type']} - {order['side'].upper()}")
    
    except Exception as e:
        logging.error(f"[CANCEL ERROR] فشل إلغاء الأوامر: {e}")
        
def show_pending_orders():
    """عرض قائمة الأوامر غير المنفذة"""
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        if not open_orders:
            logging.info("لا توجد أوامر غير منفذة")
            return []

        pending_orders = []
        for order in open_orders:
            pending_orders.append({
                'id': order['id'],
                'type': order['type'],
                'side': order['side'],
                'price': order['price'],
                'amount': order['amount'],
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            })
            logging.info(f"[PENDING ORDER] {order['side'].upper()} - {order['amount']} @ {order['price']}")
        return pending_orders

    except Exception as e:
        logging.error(f"[FETCH ORDERS ERROR] {e}")
        return []

def execute_limit_trade(signal):
    """
    تنفيذ أمر بيع/شراء بسعر محدد مع إدارة للمخاطر
    signal: 1 (شراء) أو -1 (بيع)
    """
    global open_trade, trade_history

    # --- إدارة المخاطر ---
    if open_trade['is_open']:
        logging.warning("صفقة مفتوحة بالفعل، لا يمكن فتح أخرى")
        return

    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        balance = exchange.fetch_balance()

        # --- التحقق من الرصيد ---
        if signal == 1 and not open_trade['is_open']:  # شراء
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            if usdt_balance < investment_usdt:
                logging.warning(f"رصيد USDT غير كافٍ: {usdt_balance:.2f} < {investment_usdt}")
                return

            limit_price = current_price * 0.995  # سعر أقل قليلًا من السوق
            amount = investment_usdt / limit_price

        elif signal == -1 and open_trade['is_open']:  # بيع
            btc_balance = balance.get('BTC', {}).get('free', 0)
            amount = open_trade['amount']
            if btc_balance < amount:
                logging.warning(f"رصيد BTC غير كافٍ: {btc_balance:.8f} < {amount:.8f}")
                return

            limit_price = current_price * 1.005  # سعر أعلى قليلًا

        else:
            return

        # --- تنفيذ الأمر ---
        if signal == 1:
            order = exchange.create_limit_buy_order(symbol, amount, limit_price)
            open_trade = {
                'is_open': True,
                'buy_price': limit_price,
                'amount': amount,
                'buy_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            trade_history.append({
                'type': 'BUY',
                'price': limit_price,
                'amount': amount,
                'time': open_trade['buy_time'],
                'status': 'pending'
            })
            logging.info(f"[LIMIT BUY] {amount:.8f} BTC at ${limit_price:.2f}")

        elif signal == -1:
            order = exchange.create_limit_sell_order(symbol, amount, limit_price)
            profit = (limit_price - open_trade['buy_price']) * amount
            trade_history.append({
                'type': 'SELL',
                'price': limit_price,
                'amount': amount,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'profit': profit,
                'status': 'pending'
            })
            logging.info(f"[LIMIT SELL] {amount:.8f} BTC at ${limit_price:.2f}")
            logging.info(f"Profit expected: ${profit:.2f}")
            open_trade['is_open'] = False

    except Exception as e:
        logging.error(f"[TRADE EXECUTION ERROR] {e}")
            
# === وظائف التحديث ===
def update_market_data():
    """تحديث بيانات السوق المخزنة مؤقتاً"""
    try:
        cached_market_data['balance'] = exchange.fetch_balance()
        cached_market_data['ticker'] = exchange.fetch_ticker(symbol)
        cached_market_data['ohlcv'] = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=24)  # 24 ساعة
        cached_market_data['last_updated'] = datetime.now()
        logging.info("تم تحديث بيانات السوق بنجاح")
    except Exception as e:
        logging.error(f"فشل تحديث بيانات السوق: {e}")
        
def get_live_data():
    """جلب بيانات السوق الحية (24 ساعة) مع معالجة الأخطاء"""
    try:
        # جلب بيانات أكثر من المطلوب لضمان وجود 24 نقطة صالحة
        bars = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=30)  # جلب 30 نقطة للتأمين
        
        if not bars or len(bars) < 24:
            raise ValueError(f"تم استرجاع {len(bars) if bars else 0} نقطة فقط")
            
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    except Exception as e:
        logging.error(f"فشل جلب البيانات: {e}")
        # محاولة استخدام البيانات المخزنة إذا كانت كافية
        cached_data = cached_market_data.get('ohlcv')
        if cached_data and len(cached_data) >= 24:
            return pd.DataFrame(cached_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        raise ValueError("لا توجد بيانات كافية في الذاكرة المؤقتة")


def update_prediction():
    """إصدار محسن مع إدارة أفضل للأخطاء"""
    try:
        df = get_live_data()
        X = prepare_features(df)
        cached_market_data['last_prediction'] = predict_direction()
        logging.info(f"تم تحديث التنبؤ: {cached_market_data['last_prediction']}")
    except ValueError as e:
        logging.warning(f"تحذير في التحديث: {e}")
        cached_market_data['last_prediction'] = None
    except Exception as e:
        logging.error(f"فشل تحديث التنبؤ: {e}", exc_info=True)
        cached_market_data['last_prediction'] = None

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
    
    # حساب وقت الفحص التالي (كل ساعة)
    next_check = (datetime.now() + timedelta(minutes=15 - (datetime.now().minute % 15))).strftime("%H:%M:%S")
    
    # معالجة last_updated بشكل آمن
    last_updated = cached_market_data.get('last_updated')
    if last_updated is None:
        last_updated = datetime.now()
    last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template('dashboard.html',
                        open_trade=open_trade,
                        balance=processed_balance,
                        trades=trade_history,
                        symbol=symbol,
                        current_price=current_price,
                        current_profit=current_profit,
                        last_signal=last_signal,
                        next_check=next_check,
                        last_updated=last_updated_str)
                        
@app.route('/dashboard-data')
def dashboard_data():
    try:
        # تحديث البيانات قبل الإرسال
        update_market_data()
        update_prediction()
        
        # استخدام البيانات المخزنة مؤقتاً
        balance = cached_market_data.get('balance', {})
        ticker = cached_market_data.get('ticker', {})
        
        processed_balance = {
            'free': {
                'USDT': balance.get('USDT', {}).get('free', 0) if balance else 0,
                'BTC': balance.get('BTC', {}).get('free', 0) if balance else 0
            }
        }

        current_price = ticker.get('last', 0) if ticker else 0
        current_profit = 0
        if open_trade['is_open']:
            current_profit = (current_price - open_trade['buy_price']) * open_trade['amount']
        
        last_signal = "غير معروف"
        if cached_market_data.get('last_prediction') is not None:
            last_signal = "شراء" if cached_market_data['last_prediction'] == 1 else "بيع"
        
        # وقت الفحص التالي (كل ساعة)
        next_check = (datetime.now() + timedelta(minutes=15 - (datetime.now().minute % 15))).strftime("%H:%M:%S")
        
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
        logging.error(f"Error in dashboard_data: {e}")
        return jsonify({'error': str(e)}), 500
        
# === تشغيل البوت ===
def trading_job():
    """المهمة الدورية للتداول (كل ساعة)"""
    logging.info("جارٍ التحقق من السوق (دورة الساعة)...")
    try:
        signal = predict_direction()
        if signal == 1:
            logging.info("إشارة شراء تم الكشف عنها")
        elif signal == -1:
            logging.info("إشارة بيع تم الكشف عنها")
        else:
            logging.info("لا إشارة واضحة")
        execute_limit_trade(signal)
    except Exception as e:
        logging.error(f"[ERROR] {e}")
        
def update_cache():
    """تحديث البيانات المخزنة مؤقتاً بشكل دوري"""
    while True:
        try:
            update_market_data()
            update_prediction()
            time.sleep(600)  # تحديث كل 5 دقائق
        except Exception as e:
            logging.error(f"خطأ في تحديث البيانات المخزنة: {e}")

def run_bot():
    """تشغيل البوت في الخلفية (دورة 30 دقيقة)"""
    update_market_data()
    update_prediction()

    # جدولة المهام (كل 30 دقيقة)
    schedule.every(30).minutes.do(trading_job)        # التنفيذ كل 30 دقيقة
    schedule.every(15).minutes.do(update_prediction)  # التنبؤ كل 30 دقيقة
    schedule.every(1).hours.do(update_market_data)    # تحديث البيانات كل ساعة (يمكن تقليله إذا لزم الأمر)
    schedule.every(6).hours.do(clear_unfilled_orders) # إلغاء الأوامر كل 6 ساعات

    logging.info("بدأ تشغيل بوت التداول (دورة 30 دقيقة)")
    while True:
        schedule.run_pending()
        time.sleep(1)
        
if __name__ == '__main__':
    # بدء تحديث البيانات في الخلفية
    cache_thread = threading.Thread(target=update_cache)
    cache_thread.daemon = True
    cache_thread.start()
    
    # بدء البوت في خيط منفصل
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # بدء خادم Flask
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)), use_reloader=False)
