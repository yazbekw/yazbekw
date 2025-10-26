import os
import pandas as pd
import numpy as np
import hashlib
from binance.client import Client
from binance.enums import *
import time
from datetime import datetime, timedelta
import requests
import logging
import warnings
import threading
from flask import Flask, jsonify, request
import pytz
from dotenv import load_dotenv
from functools import wraps
import secrets

warnings.filterwarnings('ignore')
load_dotenv()

# ========== الإعدادات الأساسية ==========
TRADING_SETTINGS = {
    'symbols': ["BNBUSDT", "ETHUSDT"],
    'base_trade_amount': 3,
    'leverage': 50,
    'position_size': 3 * 50,
    'max_simultaneous_trades': 1,
}

RISK_SETTINGS = {
    'atr_period': 14,
    'risk_ratio': 0.5,
    'volatility_multiplier': 1.5,
    'margin_risk_threshold': 0.7,
    'position_reduction': 0.5,
    # ⭐ إعدادات وقف الخسارة المطور
    'stop_loss_phases': {
        'PHASE_1': {'distance_ratio': 0.5, 'allocation': 0.5},  # منتصف المسافة - 50% من المركز
        'PHASE_2': {'distance_ratio': 1.0, 'allocation': 0.5}   # المسافة الكاملة - 50% المتبقية
    },
    'min_stop_distance': 0.003,  # 0.3% - الحد الأدنى للمسافة
    'max_stop_distance': 0.015,  # 1.5% - الحد الأقصى للمسافة
    'emergency_stop_ratio': 0.01,  # 1% - وقف الطوارئ إذا كسر الحد الأدنى
    'max_trade_duration_hours': 1  # ⭐ إغلاق الصفقة بعد ساعة
}

TAKE_PROFIT_LEVELS = {
    'LEVEL_1': {'target': 0.0020, 'allocation': 0.5},
    'LEVEL_2': {'target': 0.0030, 'allocation': 0.3},
    'LEVEL_3': {'target': 0.0032, 'allocation': 0.2}
}

damascus_tz = pytz.timezone('Asia/Damascus')
app = Flask(__name__)

# ========== إعدادات الأمان ==========
API_KEYS = {
    os.getenv("MANAGER_API_KEY", "manager_key_here"): "trade_manager"
}

def require_api_key(f):
    """مصادقة على الـ API"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not api_key or api_key not in API_KEYS:
            return jsonify({'success': False, 'message': 'غير مصرح بالوصول'}), 401
        return f(*args, **kwargs)
    return decorated_function

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_manager_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    """مدير إشعارات التلغرام مع معالجة الأخطاء المحسنة"""
    
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.test_connection()
    
    def test_connection(self):
        """اختبار اتصال Telegram"""
        try:
            if not self.token or not self.chat_id:
                logger.error("❌ مفاتيح Telegram غير موجودة في متغيرات البيئة")
                return False
            
            test_url = f"{self.base_url}/getMe"
            response = requests.get(test_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ اتصال Telegram نشط")
                return True
            else:
                logger.error(f"❌ فشل اختبار Telegram: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في اختبار Telegram: {e}")
            return False
    
    def send_message(self, message, message_type='info'):
        """إرسال رسالة مع معالجة أخطاء محسنة"""
        try:
            if not self.token or not self.chat_id:
                logger.warning("⚠️ مفاتيح Telegram غير متوفرة")
                return False
            
            if not message or len(message.strip()) == 0:
                logger.warning("⚠️ محاولة إرسال رسالة فارغة")
                return False
            
            # تقليم الرسالة إذا كانت طويلة جداً
            if len(message) > 4096:
                original_length = len(message)
                message = message[:4090] + "..."
                logger.warning(f"📝 تقليم الرسالة من {original_length} إلى 4096 حرف")
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            logger.info(f"📨 جاري إرسال إشعار Telegram...")
            
            response = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
            
            if response.status_code == 200:
                logger.info("✅ تم إرسال إشعار Telegram بنجاح")
                return True
            else:
                error_msg = f"⚠️ فشل إرسال إشعار Telegram: {response.status_code} - {response.text}"
                logger.warning(error_msg)
                return False
                
        except requests.exceptions.Timeout:
            logger.error("⏰ timeout في إرسال رسالة Telegram")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("🔌 خطأ اتصال في إرسال رسالة Telegram")
            return False
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع في إرسال رسالة تلغرام: {e}")
            return False

class DynamicStopLoss:
    """نظام وقف الخسارة الديناميكي مع مرحلتين"""
    
    def __init__(self, atr_period=14, risk_ratio=0.5, stop_loss_phases=None, 
                 min_stop_distance=0.003, max_stop_distance=0.015):
        self.atr_period = atr_period
        self.risk_ratio = risk_ratio
        self.stop_loss_phases = stop_loss_phases or {
            'PHASE_1': {'distance_ratio': 0.5, 'allocation': 0.5},
            'PHASE_2': {'distance_ratio': 1.0, 'allocation': 0.5}
        }
        self.min_stop_distance = min_stop_distance
        self.max_stop_distance = max_stop_distance
    
    def calculate_atr(self, df):
        """حساب Average True Range مع معالجة الأخطاء"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.atr_period).mean()
            return atr
        except Exception as e:
            logger.error(f"❌ خطأ في حساب ATR: {e}")
            return pd.Series([df['close'].iloc[-1] * 0.01] * len(df))
    
    def calculate_support_resistance(self, df):
        """حساب مستويات الدعم والمقاومة"""
        try:
            df_with_atr = df.copy()
            df_with_atr['atr'] = self.calculate_atr(df_with_atr)
            
            # استخدام قيمة افتراضية إذا فشل حساب ATR
            if df_with_atr['atr'].isna().all() or df_with_atr['atr'].iloc[-1] == 0:
                current_price = df_with_atr['close'].iloc[-1]
                default_atr = current_price * 0.01
                df_with_atr['atr'] = default_atr
                logger.warning(f"⚠️ استخدام ATR افتراضي: {default_atr:.4f}")
            
            # حساب الدعم والمقاومة
            df_with_atr['resistance'] = df_with_atr['high'].rolling(20, min_periods=1).max()
            df_with_atr['support'] = df_with_atr['low'].rolling(20, min_periods=1).min()
            
            # ملء القيم NaN
            df_with_atr['resistance'].fillna(method='bfill', inplace=True)
            df_with_atr['support'].fillna(method='bfill', inplace=True)
            df_with_atr['atr'].fillna(method='bfill', inplace=True)
            
            return df_with_atr
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الدعم/المقاومة: {e}")
            df_default = df.copy()
            current_price = df['close'].iloc[-1]
            df_default['atr'] = current_price * 0.01
            df_default['resistance'] = current_price * 1.02
            df_default['support'] = current_price * 0.98
            return df_default
    
    def calculate_dynamic_stop_loss(self, symbol, entry_price, direction, df):
        """حساب وقف الخسارة مع مرحلتين وحدود دنيا/قصوى"""
        try:
            current_atr = df['atr'].iloc[-1] if not df['atr'].isna().iloc[-1] else entry_price * 0.01
            
            # حساب وقف الخسارة الأساسي
            if direction == 'LONG':
                support_level = df['support'].iloc[-1]
                base_stop_loss = support_level - (current_atr * self.risk_ratio)
                
                # التأكد من الحد الأدنى والأقصى للمسافة
                min_stop_price = entry_price * (1 - self.max_stop_distance)
                max_stop_price = entry_price * (1 - self.min_stop_distance)
                
                base_stop_loss = max(base_stop_loss, min_stop_price)
                base_stop_loss = min(base_stop_loss, max_stop_price)
                
            else:  # SHORT
                resistance_level = df['resistance'].iloc[-1]
                base_stop_loss = resistance_level + (current_atr * self.risk_ratio)
                
                # التأكد من الحد الأدنى والأقصى للمسافة
                min_stop_price = entry_price * (1 + self.min_stop_distance)
                max_stop_price = entry_price * (1 + self.max_stop_distance)
                
                base_stop_loss = min(base_stop_loss, max_stop_price)
                base_stop_loss = max(base_stop_loss, min_stop_price)
            
            # حساب مستويين لوقف الخسارة
            stop_loss_levels = {}
            for phase, config in self.stop_loss_phases.items():
                if direction == 'LONG':
                    # المسافة من سعر الدخول إلى وقف الخسارة الأساسي
                    full_distance = entry_price - base_stop_loss
                    phase_distance = full_distance * config['distance_ratio']
                    phase_stop_loss = entry_price - phase_distance
                    
                    # التأكد من أن الوقف ليس قريباً جداً
                    min_phase_distance = entry_price * self.min_stop_distance
                    if (entry_price - phase_stop_loss) < min_phase_distance:
                        phase_stop_loss = entry_price - min_phase_distance
                    
                else:  # SHORT
                    full_distance = base_stop_loss - entry_price
                    phase_distance = full_distance * config['distance_ratio']
                    phase_stop_loss = entry_price + phase_distance
                    
                    min_phase_distance = entry_price * self.min_stop_distance
                    if (phase_stop_loss - entry_price) < min_phase_distance:
                        phase_stop_loss = entry_price + min_phase_distance
                
                stop_loss_levels[phase] = {
                    'price': phase_stop_loss,
                    'distance_ratio': config['distance_ratio'],
                    'allocation': config['allocation'],
                    'quantity': None
                }
            
            logger.info(f"💰 وقف الخسارة لـ {symbol}:")
            for phase, level in stop_loss_levels.items():
                distance_pct = abs(entry_price - level['price']) / entry_price * 100
                logger.info(f"   {phase}: {level['price']:.4f} (المسافة: {distance_pct:.2f}%)")
            
            return stop_loss_levels
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب وقف الخسارة: {e}")
            # استخدام قيم افتراضية آمنة في حالة الخطأ
            return self.get_default_stop_loss(symbol, entry_price, direction)
    
    def get_default_stop_loss(self, symbol, entry_price, direction):
        """قيم افتراضية آمنة لوقف الخسارة في حالة الخطأ"""
        default_levels = {}
        
        for phase, config in self.stop_loss_phases.items():
            if direction == 'LONG':
                # استخدام الحد الأدنى للمسافة كافتراضي آمن
                stop_price = entry_price * (1 - self.min_stop_distance)
            else:
                stop_price = entry_price * (1 + self.min_stop_distance)
            
            default_levels[phase] = {
                'price': stop_price,
                'distance_ratio': config['distance_ratio'],
                'allocation': config['allocation'],
                'quantity': None
            }
        
        logger.warning(f"⚠️ استخدام وقف خسارة افتراضي آمن لـ {symbol}")
        return default_levels

class DynamicTakeProfit:
    """نظام جني الأرباح الديناميكي"""
    
    def __init__(self, base_levels=None, volatility_multiplier=1.5):
        self.base_levels = base_levels or TAKE_PROFIT_LEVELS
        self.volatility_multiplier = volatility_multiplier
    
    def calculate_dynamic_take_profit(self, symbol, entry_price, direction, df):
        """حساب جني الأرباح الديناميكي مع تعديل التقلب"""
        try:
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].isna().iloc[-1] else 0
            current_close = df['close'].iloc[-1]
            
            take_profit_levels = {}
            
            for level, config in self.base_levels.items():
                base_target = config['target']
                
                if current_atr > 0 and current_close > 0:
                    atr_ratio = current_atr / current_close
                    volatility_factor = 1 + (atr_ratio * self.volatility_multiplier)
                    adjusted_target = base_target * volatility_factor
                else:
                    adjusted_target = base_target
                
                if direction == 'LONG':
                    tp_price = entry_price * (1 + adjusted_target)
                else:
                    tp_price = entry_price * (1 - adjusted_target)
                
                take_profit_levels[level] = {
                    'price': tp_price,
                    'target_percent': adjusted_target * 100,
                    'allocation': config['allocation'],
                    'quantity': None
                }
            
            tp_info = [f'{level}: {config["price"]:.4f}' for level, config in take_profit_levels.items()]
            logger.info(f"🎯 جني الأرباح لـ {symbol}: {tp_info}")
            return take_profit_levels
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب جني الأرباح: {e}")
            default_levels = {}
            for level, config in self.base_levels.items():
                if direction == 'LONG':
                    tp_price = entry_price * (1 + config['target'])
                else:
                    tp_price = entry_price * (1 - config['target'])
                
                default_levels[level] = {
                    'price': tp_price,
                    'target_percent': config['target'] * 100,
                    'allocation': config['allocation'],
                    'quantity': None
                }
            return default_levels
    
    def calculate_partial_close_quantity(self, total_quantity, level_allocation):
        """حساب الكمية للإغلاق الجزئي"""
        return total_quantity * level_allocation

class MarginMonitor:
    """مراقبة الهامش وتعديل المخاطرة"""
    
    def __init__(self, risk_threshold=0.7, position_reduction=0.5):
        self.risk_threshold = risk_threshold
        self.position_reduction = position_reduction
    
    def check_margin_health(self, client):
        """فحص صحة الهامش - مصححة"""
        try:
            account_info = client.futures_account()
        
            total_wallet_balance = float(account_info['totalWalletBalance'])
            available_balance = float(account_info['availableBalance'])
            total_margin_balance = float(account_info['totalMarginBalance'])
        
            if total_wallet_balance > 0:
                margin_used = total_wallet_balance - available_balance
                margin_ratio = margin_used / total_wallet_balance
            
                return {
                    'total_wallet_balance': total_wallet_balance,
                    'available_balance': available_balance,
                    'total_margin_balance': total_margin_balance,
                    'margin_used': margin_used,
                    'margin_ratio': margin_ratio,
                    'is_risk_high': margin_ratio > self.risk_threshold
                }
            return None
        
        except Exception as e:
            logger.error(f"❌ خطأ في فحص الهامش: {e}")
            return None

class CompleteTradeManager:
    """البوت الرئيسي لإدارة الصفقات"""
    
    def __init__(self, client, notifier):
        self.client = client
        self.notifier = notifier
        self.stop_loss_manager = DynamicStopLoss(
            atr_period=RISK_SETTINGS['atr_period'],
            risk_ratio=RISK_SETTINGS['risk_ratio'],
            stop_loss_phases=RISK_SETTINGS['stop_loss_phases'],
            min_stop_distance=RISK_SETTINGS['min_stop_distance'],
            max_stop_distance=RISK_SETTINGS['max_stop_distance']
        )
        self.take_profit_manager = DynamicTakeProfit()
        self.margin_monitor = MarginMonitor()
        self.managed_trades = {}
        self.performance_stats = {
            'total_trades_managed': 0,
            'profitable_trades': 0,
            'stopped_trades': 0,
            'take_profit_hits': 0,
            'timeout_trades': 0,  # ⭐ صفقات انتهى وقتها
            'total_pnl': 0
        }
        self.last_heartbeat = datetime.now(damascus_tz)  # ⭐ تتبع آخر نبضة
        self.symbols_info = {}  # ⭐ تخزين معلومات الرموز لدقة الكميات
        self.price_cache = {}  # ⭐ كاش للأسعار
        self.cache_timeout = 30  # ⭐ ثواني قبل انتهاء صلاحية الكاش
        self.last_api_call = {}  # ⭐ تتبع آخر طلب API
    
    def get_symbol_info(self, symbol):
        """الحصول على معلومات الرمز من Binance مرة واحدة وتخزينها"""
        try:
            if symbol not in self.symbols_info:
                exchange_info = self.client.futures_exchange_info()
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        self.symbols_info[symbol] = symbol_info
                        logger.info(f"✅ تم تحميل معلومات الرمز: {symbol}")
                        break
            return self.symbols_info.get(symbol)
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على معلومات الرمز {symbol}: {e}")
            return None

    def get_current_position(self, symbol):
        """الحصول على المركز الحالي من Binance مباشرة"""
        try:
            positions = self.client.futures_account()['positions']
            for position in positions:
                if position['symbol'] == symbol:
                    position_amt = float(position['positionAmt'])
                    return {
                        'position_amt': position_amt,
                        'entry_price': float(position['entryPrice']),
                        'unrealized_pnl': float(position['unrealizedProfit'])
                    }
            return None
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على المركز لـ {symbol}: {e}")
            return None
    
    def adjust_quantity_precision(self, symbol, quantity):
        """تصحيح دقة الكمية حسب متطلبات Binance"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                # استخدام قيم افتراضية آمنة إذا فشل الحصول على المعلومات
                return round(quantity, 3)
            
            # البحث عن filter LOT_SIZE
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    
                    # حساب الدقة المناسبة
                    precision = 0
                    while step_size < 1:
                        step_size *= 10
                        precision += 1
                    
                    # تقريب الكمية للدقة المطلوبة
                    adjusted_quantity = round(quantity - (quantity % float(f['stepSize'])), precision)
                    logger.info(f"📏 تصحيح كمية {symbol}: {quantity:.6f} -> {adjusted_quantity:.6f} (دقة: {precision})")
                    return adjusted_quantity
            
            return round(quantity, 3)
            
        except Exception as e:
            logger.error(f"❌ خطأ في تصحيح دقة الكمية لـ {symbol}: {e}")
            return round(quantity, 3)
    
    def validate_quantity(self, symbol, quantity):
        """التحقق من أن الكمية تفي بالحد الأدنى المطلوب"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return quantity > 0  # تحقق أساسي فقط
            
            for f in symbol_info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    if quantity < min_qty:
                        logger.warning(f"⚠️ الكمية {quantity} أقل من الحد الأدنى {min_qty} لـ {symbol}")
                        return False
                    return True
            
            return quantity > 0
            
        except Exception as e:
            logger.error(f"❌ خطأ في التحقق من الكمية لـ {symbol}: {e}")
            return quantity > 0
    
    def debug_active_positions(self):
        """تصحيح أخطاء رصد الصفقات"""
        try:
            positions = self.client.futures_account()['positions']
            logger.info("🔍 فحص جميع المراكز في Binance")
            
            active_count = 0
            for position in positions:
                symbol = position['symbol']
                position_amt = float(position['positionAmt'])
                entry_price = float(position['entryPrice'])
                unrealized_pnl = float(position['unrealizedProfit'])
                
                if position_amt != 0:
                    active_count += 1
                    logger.info(f"🔍 مركز نشط: {symbol} | الكمية: {position_amt} | السعر: {entry_price} | PnL: {unrealized_pnl}")
                
                if symbol in TRADING_SETTINGS['symbols']:
                    logger.info(f"🔍 عملة مدعومة: {symbol} | الكمية: {position_amt}")
            
            logger.info(f"🔍 إجمالي المراكز النشطة: {active_count}")
            return active_count
            
        except Exception as e:
            logger.error(f"❌ خطأ في تصحيح المراكز: {e}")
            return 0
    
    def get_price_data(self, symbol, interval='15m', limit=20):  # ⭐ تقليل الحد من 50 إلى 20
        """الحصول على بيانات السعر مع تقليل الطلبات"""
        try:
            # ⭐ تأخير عشوائي لتجنب الطلبات المتزامنة
            time.sleep(np.random.uniform(1, 2))
        
            klines = self.client.futures_klines(
                symbol=symbol, 
                interval=interval, 
                limit=limit
            )
        
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
        
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
        
            return df
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على بيانات السعر لـ {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """الحصول على السعر الحالي مع نظام كاش"""
        try:
            current_time = time.time()
            
            # ⭐ التحقق من الكاش أولاً
            if (symbol in self.price_cache and 
                current_time - self.price_cache[symbol]['timestamp'] < self.cache_timeout):
                return self.price_cache[symbol]['price']
            
            # ⭐ التحقق من آخر طلب API لتجنب Rate Limit
            if symbol in self.last_api_call:
                time_since_last_call = current_time - self.last_api_call[symbol]
                if time_since_last_call < 1:  # طلب واحد في الثانية كحد أدنى
                    time.sleep(1 - time_since_last_call)
            
            # ⭐ تأخير عشوائي إضافي
            time.sleep(np.random.uniform(0.5, 1.0))
            
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            
            # ⭐ تحديث الكاش
            self.price_cache[symbol] = {
                'price': price,
                'timestamp': current_time
            }
            
            # ⭐ تحديث آخر طلب API
            self.last_api_call[symbol] = current_time
            
            return price
            
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على سعر {symbol}: {e}")
            return None
    
    def get_active_positions_from_binance(self):
        """الحصول على الصفقات النشطة من Binance"""
        try:
            positions = self.client.futures_account()['positions']
            active_positions = []
            
            logger.info(f"🔍 جاري فحص {len(positions)} مركز")
            
            for position in positions:
                symbol = position['symbol']
                position_amt = float(position['positionAmt'])
                
                if symbol in TRADING_SETTINGS['symbols']:
                    logger.info(f"🔍 فحص {symbol}: الكمية = {position_amt}")
                
                if position_amt != 0 and symbol in TRADING_SETTINGS['symbols']:
                    active_positions.append({
                        'symbol': symbol,
                        'quantity': abs(position_amt),
                        'entry_price': float(position['entryPrice']),
                        'direction': 'LONG' if position_amt > 0 else 'SHORT',
                        'leverage': int(position['leverage']),
                        'unrealized_pnl': float(position['unrealizedProfit']),
                        'position_amt': position_amt
                    })
                    logger.info(f"✅ تم رصد صفقة نشطة: {symbol} | الاتجاه: {'LONG' if position_amt > 0 else 'SHORT'} | الكمية: {abs(position_amt)}")
            
            logger.info(f"✅ تم العثور على {len(active_positions)} صفقة نشطة")
            return active_positions
            
        except Exception as e:
            logger.error(f"❌ خطأ في الحصول على الصفقات من Binance: {e}")
            return []
    
    def sync_with_binance_positions(self):
        """مزامنة الصفقات مع Binance مع تقليل الطلبات"""
        try:
            # ⭐ تأخير عشوائي قبل البدء
            time.sleep(np.random.uniform(2, 5))
        
            # ⭐ استخدام التصحيح فقط عند الحاجة (ليس في كل مزامنة)
            if len(self.managed_trades) == 0:
                self.debug_active_positions()
        
            active_positions = self.get_active_positions_from_binance()
            current_managed = set(self.managed_trades.keys())
            binance_symbols = {pos['symbol'] for pos in active_positions}
        
            logger.info(f"🔄 المزامنة: {len(active_positions)} صفقة في Binance, {len(current_managed)} صفقة مدارة")
        
            # ⭐ تقليل التسجيل التفصيلي
            added_count = 0
            for position in active_positions:
                if position['symbol'] not in current_managed:
                    logger.info(f"🔄 إضافة صفقة جديدة: {position['symbol']}")
                
                    df = self.get_price_data(position['symbol'])
                    if df is not None and not df.empty:
                        success = self.manage_new_trade(position)
                        if success:
                            added_count += 1
                            self.send_trade_discovery_notification(position)
                        else:
                            logger.error(f"❌ فشل بدء إدارة {position['symbol']}")
                
                    # ⭐ تأخير بين إضافة الصفقات الجديدة
                    time.sleep(3)
        
            removed_count = 0
            for symbol in list(current_managed):
                if symbol not in binance_symbols:
                    if symbol in self.managed_trades:
                        del self.managed_trades[symbol]
                        removed_count += 1
        
            logger.info(f"🔄 انتهت المزامنة: أضيف {added_count}، أزيل {removed_count}")
            return len(active_positions)
        
        except Exception as e:
            logger.error(f"❌ خطأ في المزامنة مع Binance: {e}")
            return 0
    
    def manage_new_trade(self, trade_data):
        """بدء إدارة صفقة جديدة مع تصحيح دقة الكميات"""
        symbol = trade_data['symbol']
        
        logger.info(f"🔄 بدء إدارة صفقة جديدة: {symbol}")
        
        df = self.get_price_data(symbol)
        if df is None or df.empty:
            logger.error(f"❌ لا يمكن إدارة {symbol} - بيانات السعر غير متوفرة")
            return False
        
        try:
            # ⭐ تحميل معلومات الرمز أولاً
            self.get_symbol_info(symbol)
            
            df = self.stop_loss_manager.calculate_support_resistance(df)
            
            stop_loss_levels = self.stop_loss_manager.calculate_dynamic_stop_loss(
                symbol, trade_data['entry_price'], trade_data['direction'], df
            )
            
            take_profit_levels = self.take_profit_manager.calculate_dynamic_take_profit(
                symbol, trade_data['entry_price'], trade_data['direction'], df
            )
            
            # حساب الكميات لكل مستوى مع تصحيح الدقة
            total_quantity = trade_data['quantity']
            
            for phase, config in stop_loss_levels.items():
                raw_quantity = total_quantity * config['allocation']
                config['quantity'] = self.adjust_quantity_precision(symbol, raw_quantity)
                logger.info(f"📊 وقف خسارة {phase}: {raw_quantity:.6f} -> {config['quantity']:.6f}")
            
            for level, config in take_profit_levels.items():
                raw_quantity = self.take_profit_manager.calculate_partial_close_quantity(
                    total_quantity, config['allocation']
                )
                config['quantity'] = self.adjust_quantity_precision(symbol, raw_quantity)
                logger.info(f"📊 جني أرباح {level}: {raw_quantity:.6f} -> {config['quantity']:.6f}")
            
            # ⭐ إضافة وقت انتهاء الصفقة (ساعة من الآن)
            trade_expiry = datetime.now(damascus_tz) + timedelta(hours=RISK_SETTINGS['max_trade_duration_hours'])
            
            self.managed_trades[symbol] = {
                **trade_data,
                'stop_loss_levels': stop_loss_levels,
                'take_profit_levels': take_profit_levels,
                'closed_stop_levels': [],
                'closed_tp_levels': [],
                'last_update': datetime.now(damascus_tz),
                'status': 'managed',
                'management_start': datetime.now(damascus_tz),
                'trade_expiry': trade_expiry,
                'trade_discovered_at': datetime.now(damascus_tz)
            }
            
            self.performance_stats['total_trades_managed'] += 1
            
            # إرسال إشعار بدء الإدارة مع تفاصيل الوقف الجديد
            self.send_management_start_notification(symbol)
            return True
            
        except Exception as e:
            logger.error(f"❌ فشل إدارة الصفقة {symbol}: {e}")
            return False
    
    def check_managed_trades(self):
        """فحص الصفقات المدارة مع تقليل الطلبات"""
        closed_trades = []
    
        # ⭐ إضافة تأخير بين فحص كل صفقة
        for symbol, trade in list(self.managed_trades.items()):
            try:
                # ⭐ التحقق أولاً من أن المركز لا يزال موجوداً في Binance
                current_position = self.get_current_position(symbol)
                if not current_position or current_position['position_amt'] == 0:
                    logger.info(f"🔄 المركز أصبح صفراً لـ {symbol} - إزالة من الإدارة")
                    if symbol in self.managed_trades:
                        del self.managed_trades[symbol]
                    closed_trades.append(symbol)
                    continue
            
                # ⭐ الحصول على السعر الحالي مع معالجة الأخطاء
                current_price = self.get_current_price(symbol)
                if not current_price:
                    time.sleep(1)  # تأخير بسيط قبل المحاولة التالية
                    continue
            
                # 1. فحص انتهاء وقت الصفقة
                if self.check_trade_timeout(symbol):
                    closed_trades.append(symbol)
                    continue
            
                # 2. فحص وقف الخسارة
                if self.check_stop_loss(symbol, current_price):
                    closed_trades.append(symbol)
                    continue
            
                # 3. فحص جني الأرباح
                self.check_take_profits(symbol, current_price)
            
                # 4. تحديث المستويات الديناميكية كل ساعة (بدلاً من كل فحص)
                if (datetime.now(damascus_tz) - trade['last_update']).seconds > 3600:
                    self.update_dynamic_levels(symbol)
            
                # ⭐ تأخير بين فحص الصفقات لتقليل الطلبات
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"❌ خطأ في فحص الصفقة {symbol}: {e}")
                time.sleep(5)  # تأخير أطول عند الأخطاء
    
        return closed_trades
    
    def check_trade_timeout(self, symbol):
        """⭐ فحص انتهاء وقت الصفقة (ساعة من الفتح)"""
        try:
            trade = self.managed_trades[symbol]
            current_time = datetime.now(damascus_tz)
            
            if current_time >= trade['trade_expiry']:
                logger.warning(f"⏰ انتهاء وقت الصفقة {symbol} - الإغلاق الإجباري")
                
                success, message = self.close_entire_trade(symbol, "انتهاء وقت الصفقة (ساعة)")
                if success:
                    self.performance_stats['timeout_trades'] += 1
                    
                    # إرسال إشعار انتهاء الوقت
                    self.send_timeout_notification(trade)
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في فحص وقت الصفقة {symbol}: {e}")
            return False
    
    def check_stop_loss(self, symbol, current_price):
        """فحص وقف الخسارة مع المرحلتين"""
        trade = self.managed_trades[symbol]
        
        for phase, config in trade['stop_loss_levels'].items():
            if phase in trade['closed_stop_levels']:
                continue
            
            should_close = False
            reason = f"وقف خسارة {phase}"
            
            if trade['direction'] == 'LONG' and current_price <= config['price']:
                should_close = True
            elif trade['direction'] == 'SHORT' and current_price >= config['price']:
                should_close = True
            
            if should_close:
                logger.info(f"🚨 ينبغي إغلاق جزء من {symbol} بسبب {reason}")
                success = self.close_partial_stop_loss(symbol, phase, config)
                if success:
                    trade['closed_stop_levels'].append(phase)
                    logger.info(f"✅ تم إغلاق وقف الخسارة {phase} لـ {symbol}")
                    
                    # إرسال إشعار وقف الخسارة الجزئي
                    self.send_stop_loss_notification(trade, phase, current_price, config)
                    
                    # إذا كانت جميع مستويات الوقف قد تم تفعيلها، أغلق الصفقة بالكامل
                    if len(trade['closed_stop_levels']) == len(trade['stop_loss_levels']):
                        self.close_entire_trade(symbol, "تفعيل جميع مستويات وقف الخسارة")
                        self.performance_stats['stopped_trades'] += 1
                    return True
        
        return False
    
    def check_take_profits(self, symbol, current_price):
        """فحص مستويات جني الأرباح"""
        trade = self.managed_trades[symbol]
        
        for level, config in trade['take_profit_levels'].items():
            if level in trade['closed_tp_levels']:
                continue
            
            should_close = False
            if trade['direction'] == 'LONG' and current_price >= config['price']:
                should_close = True
            elif trade['direction'] == 'SHORT' and current_price <= config['price']:
                should_close = True
            
            if should_close:
                success = self.close_partial_trade(symbol, level, config)
                if success:
                    trade['closed_tp_levels'].append(level)
                    self.performance_stats['take_profit_hits'] += 1
                    
                    # إرسال إشعار جني الأرباح
                    self.send_take_profit_notification(trade, level, current_price)
                    
                    if len(trade['closed_tp_levels']) == len(trade['take_profit_levels']):
                        self.close_entire_trade(symbol, "تم جني جميع مستويات الربح")
                        self.performance_stats['profitable_trades'] += 1
    
    def close_partial_stop_loss(self, symbol, phase, config):
        """إغلاق جزئي بسبب وقف الخسارة مع تصحيح الدقة والتحقق من المركز"""
        try:
            if symbol not in self.managed_trades:
                return False
        
            trade = self.managed_trades[symbol]
            quantity = config['quantity']
        
            # ⭐ التحقق من المركز الحقيقي أولاً
            current_position = self.get_current_position(symbol)
            if not current_position or current_position['position_amt'] == 0:
                logger.warning(f"⚠️ لا يوجد مركز نشط لـ {symbol}")
                if symbol in self.managed_trades:
                    del self.managed_trades[symbol]
                return False
        
            # ⭐ التحقق من أن الكمية لا تتجاوز المركز المتبقي
            remaining_position = abs(current_position['position_amt'])
            if quantity > remaining_position:
                logger.warning(f"⚠️ ضبط الكمية لتتناسب مع المركز المتبقي: {quantity} -> {remaining_position}")
                quantity = remaining_position
        
            # ⭐ التحقق من الحد الأدنى للكمية
            adjusted_quantity = self.adjust_quantity_precision(symbol, quantity)
            if not self.validate_quantity(symbol, adjusted_quantity):
                logger.error(f"❌ كمية غير صالحة لـ {symbol}: {adjusted_quantity}")
                return False
        
            logger.info(f"🔧 جاري إغلاق وقف خسارة جزئي لـ {symbol} - المرحلة {phase}")
            logger.info(f"📊 الكمية المصححة: {adjusted_quantity:.6f}")
            logger.info(f"📊 المركز المتبقي: {remaining_position:.6f}")
        
            # تحديد اتجاه الإغلاق بناءً على المركز الفعلي
            if current_position['position_amt'] > 0:  # LONG
                side = 'SELL'
            else:  # SHORT
                side = 'BUY'
        
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_quantity,
                reduceOnly=True
            )
        
            if order:
                logger.info(f"✅ إغلاق وقف خسارة جزئي لـ {symbol} - المرحلة {phase}: {adjusted_quantity:.6f}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"❌ خطأ في إغلاق وقف الخسارة الجزئي لـ {symbol}: {e}")
            return False

    def close_partial_trade(self, symbol, level, config):
        """إغلاق جزئي للصفقة مع تصحيح الدقة والتحقق من المركز"""
        try:
            if symbol not in self.managed_trades:
                return False
        
            trade = self.managed_trades[symbol]
            quantity = config['quantity']
        
            # ⭐ التحقق من المركز الحقيقي أولاً
            current_position = self.get_current_position(symbol)
            if not current_position or current_position['position_amt'] == 0:
                logger.warning(f"⚠️ لا يوجد مركز نشط لـ {symbol}")
                if symbol in self.managed_trades:
                    del self.managed_trades[symbol]
                return False
        
            # ⭐ التحقق من أن الكمية لا تتجاوز المركز المتبقي
            remaining_position = abs(current_position['position_amt'])
            if quantity > remaining_position:
                logger.warning(f"⚠️ ضبط الكمية لتتناسب مع المركز المتبقي: {quantity} -> {remaining_position}")
                quantity = remaining_position
        
            # ⭐ التحقق من الحد الأدنى للكمية
            adjusted_quantity = self.adjust_quantity_precision(symbol, quantity)
            if not self.validate_quantity(symbol, adjusted_quantity):
                logger.error(f"❌ كمية غير صالحة لـ {symbol}: {adjusted_quantity}")
                return False
        
            logger.info(f"🔧 جاري إغلاق جزئي لـ {symbol} - المستوى {level}")
            logger.info(f"📊 الكمية المصححة: {adjusted_quantity:.6f}")
            logger.info(f"📊 المركز المتبقي: {remaining_position:.6f}")
        
            # تحديد اتجاه الإغلاق بناءً على المركز الفعلي
            if current_position['position_amt'] > 0:  # LONG
                side = 'SELL'
            else:  # SHORT
                side = 'BUY'
        
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=adjusted_quantity,
                reduceOnly=True
            )
        
            if order:
                logger.info(f"✅ جني رباح جزئي لـ {symbol} - المستوى {level}: {adjusted_quantity:.6f}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"❌ خطأ في الجني الجزئي لـ {symbol}: {e}")
            return False
    
    def close_entire_trade(self, symbol, reason):
        """إغلاق كامل للصفقة مع تصحيح الدقة والتحقق من المركز"""
        try:
            if symbol not in self.managed_trades:
                return False, "الصفقة غير موجودة في الإدارة"
        
            trade = self.managed_trades[symbol]
        
            # ⭐ الحصول على المركز الحقيقي من Binance بدلاً من الاعتماد على الحسابات الداخلية فقط
            current_position = self.get_current_position(symbol)
            if not current_position:
                logger.warning(f"⚠️ لا يوجد مركز نشط لـ {symbol} - إزالة من الإدارة")
                if symbol in self.managed_trades:
                    del self.managed_trades[symbol]
                return False, "لا يوجد مركز نشط"
        
            position_amt = current_position['position_amt']
            if position_amt == 0:
                logger.warning(f"⚠️ المركز صفر لـ {symbol} - إزالة من الإدارة")
                if symbol in self.managed_trades:
                    del self.managed_trades[symbol]
                return False, "المركز صفر"
        
            # استخدام الكمية الحقيقية من Binance
            remaining_quantity = abs(position_amt)
        
            if remaining_quantity > 0:
                # ⭐ تصحيح دقة الكمية المتبقية
                adjusted_quantity = self.adjust_quantity_precision(symbol, remaining_quantity)
            
                if not self.validate_quantity(symbol, adjusted_quantity):
                    logger.error(f"❌ كمية غير صالحة لـ {symbol}: {adjusted_quantity}")
                    return False, "كمية غير صالحة"
            
                logger.info(f"🔧 جاري إغلاق كامل لـ {symbol}")
                logger.info(f"📊 الكمية الفعلية: {remaining_quantity:.6f}")
                logger.info(f"📊 الكمية المصححة: {adjusted_quantity:.6f}")
                logger.info(f"📊 الاتجاه: {trade['direction']}")
            
                # تحديد اتجاه الإغلاق بناءً على المركز الفعلي
                if position_amt > 0:  # LONG
                    side = 'SELL'
                else:  # SHORT
                    side = 'BUY'
            
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=adjusted_quantity,
                    reduceOnly=True
                )
            
                if order:
                    logger.info(f"✅ إغلاق كامل ناجح لـ {symbol}: {reason}")
                
                    # ⭐ التحقق من أن المركز أصبح صفراً بعد الإغلاق
                    time.sleep(2)  # انتظار قصير للتأكد من التنفيذ
                    final_check = self.get_current_position(symbol)
                    if final_check and final_check['position_amt'] == 0:
                        if symbol in self.managed_trades:
                            del self.managed_trades[symbol]
                        logger.info(f"✅ تأكيد إغلاق {symbol} - المركز أصبح صفراً")
                    else:
                        logger.warning(f"⚠️ تحذير: المركز قد لا يكون مغلقاً بالكامل لـ {symbol}")
                
                    return True, "تم الإغلاق بنجاح"
                else:
                    return False, "فشل إنشاء الأمر"
        
            return False, "لا توجد كمية للإغلاق"
        
        except Exception as e:
            logger.error(f"❌ خطأ في الإغلاق الكامل لـ {symbol}: {e}")
            return False, str(e)
    
    def calculate_pnl_percentage(self, trade, current_price):
        """حساب نسبة الربح/الخسارة"""
        if trade['direction'] == 'LONG':
            return (current_price - trade['entry_price']) / trade['entry_price'] * 100
        else:
            return (trade['entry_price'] - current_price) / trade['entry_price'] * 100
    
    def update_dynamic_levels(self, symbol):
        """تحديث مستويات وقف الخسارة وجني الأرباح"""
        if symbol not in self.managed_trades:
            return
        
        trade = self.managed_trades[symbol]
        df = self.get_price_data(symbol)
        if df is None:
            return
        
        df = self.stop_loss_manager.calculate_support_resistance(df)
        new_stop_loss_levels = self.stop_loss_manager.calculate_dynamic_stop_loss(
            symbol, trade['entry_price'], trade['direction'], df
        )
        
        # تحديث مستويات وقف الخسارة (فقط إذا كانت أفضل للصفقة)
        for phase, new_level in new_stop_loss_levels.items():
            if phase in trade['stop_loss_levels']:
                current_level = trade['stop_loss_levels'][phase]
                
                if (trade['direction'] == 'LONG' and new_level['price'] > current_level['price']) or \
                   (trade['direction'] == 'SHORT' and new_level['price'] < current_level['price']):
                    self.managed_trades[symbol]['stop_loss_levels'][phase] = new_level
                    logger.info(f"🔄 تحديث وقف الخسارة لـ {symbol} - {phase}: {new_level['price']:.4f}")
        
        self.managed_trades[symbol]['last_update'] = datetime.now(damascus_tz)
    
    def monitor_margin_risk(self):
        """مراقبة مخاطر الهامش مع تقليل التكرار"""
        try:
            # ⭐ إضافة تأخير عشوائي لتجنب الطلبات المتزامنة
            time.sleep(np.random.uniform(1, 3))
        
            margin_health = self.margin_monitor.check_margin_health(self.client)
        
            if margin_health and margin_health['is_risk_high']:
                logger.warning(f"🚨 مستوى خطورة مرتفع: {margin_health['margin_ratio']:.2%}")
            
                # إرسال تحذير الهامش
                self.send_margin_warning(margin_health)
                return True
            return False
        
        except Exception as e:
            logger.error(f"❌ خطأ في فحص الهامش: {e}")
            return False
    
    def send_heartbeat(self):
        """⭐ إرسال نبضة حياة كل ساعتين"""
        try:
            current_time = datetime.now(damascus_tz)
            
            # التحقق إذا مرت ساعتين منذ آخر نبضة
            if (current_time - self.last_heartbeat).seconds >= 7200:  # 7200 ثانية = ساعتين
                message = (
                    f"💓 <b>نبضة حياة - البوت يعمل بنجاح</b>\n"
                    f"الحالة: نشط ومستقر ✅\n"
                    f"الصفقات المدارة: {len(self.managed_trades)}\n"
                    f"إجمالي الصفقات: {self.performance_stats['total_trades_managed']}\n"
                    f"آخر تحديث: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"المنطقة الزمنية: دمشق"
                )
                
                success = self.notifier.send_message(message)
                if success:
                    self.last_heartbeat = current_time
                    logger.info("💓 تم إرسال نبضة الحياة")
                return success
            return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال نبضة الحياة: {e}")
            return False
    
    def send_trade_discovery_notification(self, trade_data):
        """⭐ إرسال إشعار اكتشاف صفقة جديدة"""
        try:
            message = (
                f"🔍 <b>تم اكتشاف صفقة جديدة</b>\n"
                f"العملة: {trade_data['symbol']}\n"
                f"الاتجاه: {trade_data['direction']}\n"
                f"الكمية: {trade_data['quantity']:.6f}\n"
                f"سعر الدخول: ${trade_data['entry_price']:.4f}\n"
                f"الرافعة: {trade_data['leverage']}x\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}\n"
                f"الحالة: بدء المراقبة التلقائية 👁️"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار الاكتشاف: {e}")
            return False
    
    def send_management_start_notification(self, symbol):
        """إرسال إشعار بدء الإدارة مع تفاصيل الوقف الجديد"""
        try:
            trade = self.managed_trades[symbol]
            
            # حساب وقت انتهاء الصفقة
            expiry_time = trade['trade_expiry'].strftime('%H:%M:%S')
            time_left = trade['trade_expiry'] - datetime.now(damascus_tz)
            hours_left = time_left.seconds // 3600
            minutes_left = (time_left.seconds % 3600) // 60
            
            message = (
                f"🔄 <b>بدء إدارة صفقة جديدة</b>\n"
                f"العملة: {symbol}\n"
                f"الاتجاه: {trade['direction']}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"الكمية: {trade['quantity']:.6f}\n"
                f"⏰ انتهاء الصفقة: {expiry_time} (بعد {hours_left}h {minutes_left}m)\n"
                f"<b>مستويات وقف الخسارة:</b>\n"
            )
            
            for phase, config in trade['stop_loss_levels'].items():
                distance_pct = abs(trade['entry_price'] - config['price']) / trade['entry_price'] * 100
                message += f"• {phase}: ${config['price']:.4f} ({distance_pct:.2f}%)\n"
            
            message += f"<b>مستويات جني الأرباح:</b>\n"
            for level, config in trade['take_profit_levels'].items():
                message += f"• {level}: ${config['price']:.4f} ({config['target_percent']:.2f}%)\n"
            
            message += f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار بدء الإدارة: {e}")
            return False
    
    def send_stop_loss_notification(self, trade, phase, current_price, config):
        """إرسال إشعار وقف الخسارة الجزئي"""
        try:
            pnl_pct = self.calculate_pnl_percentage(trade, current_price)
            pnl_emoji = "🟡"  # أصفر للوقف الجزئي
            
            message = (
                f"🛑 <b>وقف خسارة جزئي</b>\n"
                f"العملة: {trade['symbol']}\n"
                f"المرحلة: {phase}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"سعر الوقف: ${config['price']:.4f}\n"
                f"سعر السوق: ${current_price:.4f}\n"
                f"الربح/الخسارة: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"الكمية: {config['quantity']:.6f}\n"
                f"المستويات المتبقية: {len(trade['stop_loss_levels']) - len(trade['closed_stop_levels'])}\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار وقف الخسارة: {e}")
            return False
    
    def send_take_profit_notification(self, trade, level, current_price):
        """إرسال إشعار جني الأرباح"""
        try:
            config = trade['take_profit_levels'][level]
            
            message = (
                f"🎯 <b>جني أرباح جزئي</b>\n"
                f"العملة: {trade['symbol']}\n"
                f"المستوى: {level}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"سعر الجني: ${current_price:.4f}\n"
                f"الربح: {config['target_percent']:.2f}%\n"
                f"الكمية: {config['quantity']:.6f}\n"
                f"المستويات المتبقية: {len(trade['take_profit_levels']) - len(trade['closed_tp_levels'])}\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار جني الأرباح: {e}")
            return False
    
    def send_timeout_notification(self, trade):
        """⭐ إرسال إشعار انتهاء وقت الصفقة"""
        try:
            current_price = self.get_current_price(trade['symbol'])
            pnl_pct = self.calculate_pnl_percentage(trade, current_price) if current_price else 0
            pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
            
            management_duration = self.get_management_duration(trade)
            
            message = (
                f"⏰ <b>انتهاء وقت الصفقة</b>\n"
                f"العملة: {trade['symbol']}\n"
                f"الاتجاه: {trade['direction']}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"سعر الخروج: ${current_price:.4f if current_price else 'N/A'}\n"
                f"الربح/الخسارة: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"مدة الإدارة: {management_duration}\n"
                f"السبب: انتهاء الوقت المحدد (ساعة)\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار انتهاء الوقت: {e}")
            return False
    
    def send_trade_closed_notification(self, trade, current_price, reason, pnl_pct):
        """إرسال إشعار إغلاق الصفقة"""
        try:
            pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
            
            message = (
                f"🔒 <b>إغلاق الصفقة</b>\n"
                f"العملة: {trade['symbol']}\n"
                f"الاتجاه: {trade['direction']}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"سعر الخروج: ${current_price:.4f}\n"
                f"الربح/الخسارة: {pnl_emoji} {pnl_pct:+.2f}%\n"
                f"السبب: {reason}\n"
                f"المستويات المحققة: {len(trade['closed_tp_levels'])}/{len(trade['take_profit_levels'])}\n"
                f"مدة الإدارة: {self.get_management_duration(trade)}\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار إغلاق الصفقة: {e}")
            return False
    
    def send_margin_warning(self, margin_health):
        """إرسال تحذير هامش"""
        try:
            message = (
                f"⚠️ <b>تحذير: مستوى خطورة مرتفع</b>\n"
                f"نسبة الهامش المستخدم: {margin_health['margin_ratio']:.2%}\n"
                f"الرصيد المتاح: ${margin_health['available_balance']:.2f}\n"
                f"إجمالي الرصيد: ${margin_health['total_wallet_balance']:.2f}\n"
                f"الحالة: مراقبة مستمرة ⚠️\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال تحذير الهامش: {e}")
            return False
    
    def send_performance_report(self):
        """إرسال تقرير أداء"""
        try:
            if self.performance_stats['total_trades_managed'] > 0:
                win_rate = (self.performance_stats['profitable_trades'] / self.performance_stats['total_trades_managed']) * 100
            else:
                win_rate = 0
            
            message = (
                f"📊 <b>تقرير أداء مدير الصفقات</b>\n"
                f"إجمالي الصفقات: {self.performance_stats['total_trades_managed']}\n"
                f"الصفقات الرابحة: {self.performance_stats['profitable_trades']}\n"
                f"معدل الربح: {win_rate:.1f}%\n"
                f"أرباح Take Profit: {self.performance_stats['take_profit_hits']}\n"
                f"صفقات Stop Loss: {self.performance_stats['stopped_trades']}\n"
                f"صفقات انتهى وقتها: {self.performance_stats['timeout_trades']}\n"
                f"الصفقات النشطة: {len(self.managed_trades)}\n"
                f"إجمالي PnL: {self.performance_stats['total_pnl']:.2f}%\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال تقرير الأداء: {e}")
            return False
    
    def get_management_duration(self, trade):
        """حساب مدة الإدارة"""
        duration = datetime.now(damascus_tz) - trade['management_start']
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        return f"{hours}h {minutes}m"

class TradeManagerBot:
    """الفئة الرئيسية للبوت المدير"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if TradeManagerBot._instance is not None:
            raise Exception("هذه الفئة تستخدم نمط Singleton")
        
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not all([self.api_key, self.api_secret]):
            raise ValueError("مفاتيح Binance مطلوبة")
        
        # تحقق من متغيرات Telegram
        logger.info(f"🔍 تحقق Telegram: token={'موجود' if self.telegram_token else 'مفقود'}, chat_id={'موجود' if self.telegram_chat_id else 'مفقود'}")
        
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.test_connection()
        except Exception as e:
            logger.error(f"❌ فشل تهيئة العميل: {e}")
            raise
        
        self.notifier = TelegramNotifier(self.telegram_token, self.telegram_chat_id)
        self.trade_manager = CompleteTradeManager(self.client, self.notifier)
        
        TradeManagerBot._instance = self
        logger.info("✅ تم تهيئة مدير الصفقات بنجاح")
    
    def test_connection(self):
        """اختبار الاتصال"""
        try:
            self.client.futures_time()
            logger.info("✅ اتصال Binance API نشط")
            return True
        except Exception as e:
            logger.error(f"❌ فشل الاتصال بـ Binance API: {e}")
            raise
    
    def test_telegram_connection(self):
        """اختبار اتصال Telegram"""
        try:
            test_message = "🧪 اختبار اتصال البوت - إذا رأيت هذه الرسالة، فإن الإشعارات تعمل بنجاح! ✅"
            success = self.notifier.send_message(test_message)
            
            if success:
                logger.info("✅ اختبار Telegram: نجح ✅")
            else:
                logger.error("❌ اختبار Telegram: فشل ❌")
                
            return success
        except Exception as e:
            logger.error(f"❌ خطأ في اختبار Telegram: {e}")
            return False
    
    def start_management(self):
        """بدء إدارة الصفقات"""
        try:
            # اختبار Telegram أولاً
            telegram_ok = self.test_telegram_connection()
            if not telegram_ok:
                logger.error("🚨 تحذير: إشعارات Telegram لا تعمل، لكن البوت سيستمر في العمل")
            
            margin_info = self.trade_manager.margin_monitor.check_margin_health(self.client)            
            if margin_info:
                logger.info(f"✅ نسبة الهامش: {margin_info['margin_ratio']:.2%}")
            
            self.trade_manager.debug_active_positions()
            
            active_count = self.trade_manager.sync_with_binance_positions()
            logger.info(f"🔄 بدء إدارة {active_count} صفقة نشطة")
            
            if self.notifier and telegram_ok:
                message = (
                    f"🚀 <b>بدء تشغيل مدير الصفقات المتكامل</b>\n"
                    f"الوظيفة: إدارة وقف الخسارة وجني الأرباح تلقائياً\n"
                    f"العملات المدعومة: {', '.join(TRADING_SETTINGS['symbols'])}\n"
                    f"تقنية وقف الخسارة: ديناميكي مع مرحلتين\n"
                    f"تقنية جني الأرباح: 3 مستويات مع تعديل التقلب\n"
                    f"⏰ ميزة جديدة: إغلاق تلقائي بعد ساعة\n"
                    f"💓 نبضات الحياة: كل ساعتين للتأكد من العمل\n"
                    f"📏 دقة كميات: مصححة تلقائياً لـ Binance\n"
                    f"الصفقات النشطة: {active_count}\n"
                    f"الحالة: جاهز للمراقبة ✅\n"
                    f"الوقت: {datetime.now(damascus_tz).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.notifier.send_message(message)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في بدء الإدارة: {e}")
            return False
    
    def management_loop(self):
        """حلقة الإدارة الرئيسية مع تقليل الطلبات"""
        last_report_time = datetime.now(damascus_tz)
        last_sync_time = datetime.now(damascus_tz)
        last_margin_check = datetime.now(damascus_tz)
        last_heartbeat_time = datetime.now(damascus_tz)
    
        while True:
            try:
                current_time = datetime.now(damascus_tz)
            
                # ⭐ إرسال نبضة الحياة كل ساعتين (بدلاً من كل دورة)
                if (current_time - last_heartbeat_time).seconds >= 7200:
                    self.trade_manager.send_heartbeat()
                    last_heartbeat_time = current_time
            
                # ⭐ فحص الصفقات المدارة كل 30 ثانية (بدلاً من 10)
                self.trade_manager.check_managed_trades()
            
                # ⭐ مراقبة الهامش كل 5 دقائق (بدلاً من 60 ثانية)
                if (current_time - last_margin_check).seconds >= 300:
                    self.trade_manager.monitor_margin_risk()
                    last_margin_check = current_time
            
                # ⭐ مزامنة الصفقات كل 10 دقائق (بدلاً من 5)
                if (current_time - last_sync_time).seconds >= 600:
                    self.trade_manager.sync_with_binance_positions()
                    last_sync_time = current_time
            
                # ⭐ تقرير الأداء كل 6 ساعات (يبقى كما هو)
                if (current_time - last_report_time).seconds >= 21600:
                    self.trade_manager.send_performance_report()
                    last_report_time = current_time
            
                # ⭐ زيادة وقت الانتظار بين الدورات
                time.sleep(30)  # 30 ثانية بدلاً من 10
            
            except KeyboardInterrupt:
                logger.info("⏹️ إيقاف البوت يدوياً...")
                break
            except Exception as e:
                logger.error(f"❌ خطأ في حلقة الإدارة: {e}")
                time.sleep(60)  # انتظار أطول عند الأخطاء

# ========== واجهة Flask ==========

@app.route('/')
def health_check():
    """فحص صحة البوت"""
    try:
        bot = TradeManagerBot.get_instance()
        
        status = {
            'status': 'running',
            'managed_trades': len(bot.trade_manager.managed_trades),
            'performance_stats': bot.trade_manager.performance_stats,
            'timestamp': datetime.now(damascus_tz).isoformat()
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/management/start', methods=['POST'])
@require_api_key
def start_management():
    """بدء إدارة صفقة جديدة يدوياً"""
    try:
        data = request.get_json()
        bot = TradeManagerBot.get_instance()
        
        success = bot.trade_manager.manage_new_trade(data)
        
        return jsonify({
            'success': success,
            'message': 'بدء الإدارة بنجاح' if success else 'فشل بدء الإدارة',
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/management/sync', methods=['POST'])
@require_api_key
def sync_positions():
    """مزامنة الصفقات مع Binance"""
    try:
        bot = TradeManagerBot.get_instance()
        count = bot.trade_manager.sync_with_binance_positions()
        
        return jsonify({
            'success': True,
            'message': f'تمت مزامنة {count} صفقة',
            'synced_positions': count,
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/management/status')
def get_management_status():
    """الحصول على حالة الإدارة"""
    try:
        bot = TradeManagerBot.get_instance()
        
        status = {
            'managed_trades': len(bot.trade_manager.managed_trades),
            'performance_stats': bot.trade_manager.performance_stats,
            'active_trades': list(bot.trade_manager.managed_trades.keys()),
            'timestamp': datetime.now(damascus_tz).isoformat()
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/management/close/<symbol>', methods=['POST'])
@require_api_key
def close_managed_trade(symbol):
    """إغلاق صفقة مدارة"""
    try:
        bot = TradeManagerBot.get_instance()
        
        if symbol in bot.trade_manager.managed_trades:
            success, message = bot.trade_manager.close_entire_trade(symbol, "إغلاق يدوي")
            return jsonify({'success': success, 'message': message})
        else:
            return jsonify({'success': False, 'message': 'لا توجد صفقة مدارة بهذا الرمز'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug/positions')
def debug_positions():
    """مسار تصحيح لرؤية الصفقات الحالية"""
    try:
        bot = TradeManagerBot.get_instance()
        active_positions = bot.trade_manager.get_active_positions_from_binance()
        
        return jsonify({
            'success': True,
            'active_positions': active_positions,
            'managed_trades': list(bot.trade_manager.managed_trades.keys()),
            'performance_stats': bot.trade_manager.performance_stats,
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug/telegram-test')
def debug_telegram_test():
    """اختبار Telegram"""
    try:
        bot = TradeManagerBot.get_instance()
        success = bot.test_telegram_connection()
        
        return jsonify({
            'success': success,
            'message': 'اختبار Telegram نجح' if success else 'اختبار Telegram فشل',
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/debug/heartbeat', methods=['POST'])
def debug_send_heartbeat():
    """إرسال نبضة حياة يدوياً"""
    try:
        bot = TradeManagerBot.get_instance()
        success = bot.trade_manager.send_heartbeat()
        
        return jsonify({
            'success': success,
            'message': 'تم إرسال النبضة' if success else 'فشل إرسال النبضة',
            'timestamp': datetime.now(damascus_tz).isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def run_flask_app():
    """تشغيل تطبيق Flask"""
    port = int(os.environ.get('MANAGER_PORT', 10001))
    app.run(host='0.0.0.0', port=port, debug=False)

def main():
    """الدالة الرئيسية"""
    try:
        bot = TradeManagerBot.get_instance()
        bot.start_management()
        
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        logger.info("🚀 بدء تشغيل مدير الصفقات المتكامل...")
        bot.management_loop()
                
    except Exception as e:
        logger.error(f"❌ فشل تشغيل البوت: {e}")

if __name__ == "__main__":
    main()
