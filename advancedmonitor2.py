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
    # ✅ إعدادات وقف الخسارة المزدوج الديناميكي
    'partial_stop_ratio': 0.30,      # 30% من المسافة للدعم
    'full_stop_ratio': 1.0,         # 100% من المسافة للدعم (الوقف الأصلي)
    'partial_close_ratio': 0.4,     # إغلاق 40% في المرحلة الأولى
    # ✅ إضافة الحد الأدنى لوقف الخسارة - قيم أكثر واقعية
    'min_stop_loss_pct': 0.015,     # 1.5% كحد أدنى من سعر الدخول
    'max_stop_loss_pct': 0.05       # 5% كحد أقصى من سعر الدخول
}

TAKE_PROFIT_LEVELS = {
    'LEVEL_1': {'target': 0.0025, 'allocation': 0.4},
    'LEVEL_2': {'target': 0.0035, 'allocation': 0.3},
    'LEVEL_3': {'target': 0.0050, 'allocation': 0.3}
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
    """نظام وقف الخسارة الديناميكي مع التقسيم المرحلي"""
    
    def __init__(self, atr_period=14, risk_ratio=0.5):
        self.atr_period = atr_period
        self.risk_ratio = risk_ratio
    
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
        """حساب وقف الخسارة المزدوج الديناميكي مع الحد الأدنى"""
        try:
            current_atr = df['atr'].iloc[-1] if not df.empty and not pd.isna(df['atr'].iloc[-1]) else entry_price * 0.01
        
            if direction == 'LONG':
                support_level = df['support'].iloc[-1]
            
                # ✅ حساب مستويي الوقف الأساسيين
                full_stop_loss = support_level - (current_atr * self.risk_ratio)
                partial_stop_loss = entry_price - ((entry_price - full_stop_loss) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ حساب الحد الأدنى والأقصى المطلق
                min_stop_loss = entry_price * (1 - RISK_SETTINGS['min_stop_loss_pct'])
                max_stop_loss = entry_price * (1 - RISK_SETTINGS['max_stop_loss_pct'])
            
                # ✅ التأكد من أن وقف الخسارة ليس أقرب من الحد الأدنى
                if full_stop_loss > min_stop_loss:
                    logger.info(f"🔧 تعديل وقف الخسارة للحد الأدنى: {RISK_SETTINGS['min_stop_loss_pct']*100}%")
                    full_stop_loss = min_stop_loss
                    # إعادة حساب الوقف الجزئي بناءً على الجديد
                    partial_stop_loss = entry_price - ((entry_price - full_stop_loss) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ التأكد من أن وقف الخسارة ليس أبعد من الحد الأقصى
                if full_stop_loss < max_stop_loss:
                    logger.info(f"🔧 تعديل وقف الخسارة للحد الأقصى: {RISK_SETTINGS['max_stop_loss_pct']*100}%")
                    full_stop_loss = max_stop_loss
                    # إعادة حساب الوقف الجزئي بناءً على الجديد
                    partial_stop_loss = entry_price - ((entry_price - full_stop_loss) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ حدود أمان إضافية
                full_stop_loss = min(full_stop_loss, entry_price * 0.99)   # لا يزيد عن 1% خسارة
                partial_stop_loss = min(partial_stop_loss, entry_price * 0.995)  # لا يزيد عن 0.5% خسارة
            
                # ✅ منع القيم غير المنطقية
                full_stop_loss = max(full_stop_loss, entry_price * 0.95)   # لا يقل عن 5% خسارة
                partial_stop_loss = max(partial_stop_loss, entry_price * 0.98)   # لا يقل عن 2% خسارة
            
            else:  # SHORT
                resistance_level = df['resistance'].iloc[-1]
            
                # ✅ حساب مستويي الوقف الأساسيين
                full_stop_loss = resistance_level + (current_atr * self.risk_ratio)
                partial_stop_loss = entry_price + ((full_stop_loss - entry_price) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ حساب الحد الأدنى والأقصى المطلق
                min_stop_loss = entry_price * (1 + RISK_SETTINGS['min_stop_loss_pct'])
                max_stop_loss = entry_price * (1 + RISK_SETTINGS['max_stop_loss_pct'])
            
                # ✅ التأكد من أن وقف الخسارة ليس أقرب من الحد الأدنى
                if full_stop_loss < min_stop_loss:
                    logger.info(f"🔧 تعديل وقف الخسارة للحد الأدنى: {RISK_SETTINGS['min_stop_loss_pct']*100}%")
                    full_stop_loss = min_stop_loss
                    # إعادة حساب الوقف الجزئي بناءً على الجديد
                    partial_stop_loss = entry_price + ((full_stop_loss - entry_price) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ التأكد من أن وقف الخسارة ليس أبعد من الحد الأقصى
                if full_stop_loss > max_stop_loss:
                    logger.info(f"🔧 تعديل وقف الخسارة للحد الأقصى: {RISK_SETTINGS['max_stop_loss_pct']*100}%")
                    full_stop_loss = max_stop_loss
                    # إعادة حساب الوقف الجزئي بناءً على الجديد
                    partial_stop_loss = entry_price + ((full_stop_loss - entry_price) * RISK_SETTINGS['partial_stop_ratio'])
            
                # ✅ حدود أمان إضافية
                full_stop_loss = max(full_stop_loss, entry_price * 1.01)   # لا يزيد عن 1% خسارة
                partial_stop_loss = max(partial_stop_loss, entry_price * 1.005)  # لا يزيد عن 0.5% خسارة
            
                # ✅ منع القيم غير المنطقية
                full_stop_loss = min(full_stop_loss, entry_price * 1.05)   # لا يقل عن 5% خسارة
                partial_stop_loss = min(partial_stop_loss, entry_price * 1.02)   # لا يقل عن 2% خسارة
        
            logger.info(f"💰 وقف الخسارة المزدوج لـ {symbol}: جزئي={partial_stop_loss:.4f}, كامل={full_stop_loss:.4f}")
            logger.info(f"📊 المسافات: جزئي={abs(entry_price-partial_stop_loss)/entry_price*100:.2f}%, كامل={abs(entry_price-full_stop_loss)/entry_price*100:.2f}%")
        
            return {
                'partial_stop_loss': partial_stop_loss,
                'full_stop_loss': full_stop_loss
            }
        
        except Exception as e:
            logger.error(f"❌ خطأ في حساب وقف الخسارة المزدوج: {e}")
            # قيم افتراضية مع الحد الأدنى
            if direction == 'LONG':
                min_stop = entry_price * (1 - RISK_SETTINGS.get('min_stop_loss_pct', 0.015))
                return {
                    'partial_stop_loss': min_stop * 0.998,
                    'full_stop_loss': min_stop
                }
            else:
                min_stop = entry_price * (1 + RISK_SETTINGS.get('min_stop_loss_pct', 0.015))
                return {
                    'partial_stop_loss': min_stop * 1.002,
                    'full_stop_loss': min_stop
                }

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
    """البوت الرئيسي لإدارة الصفقات مع وقف الخسارة المزدوج"""
    
    def __init__(self, client, notifier):
        self.client = client
        self.notifier = notifier
        self.stop_loss_manager = DynamicStopLoss()
        self.take_profit_manager = DynamicTakeProfit()
        self.margin_monitor = MarginMonitor()
        self.managed_trades = {}
        self.performance_stats = {
            'total_trades_managed': 0,
            'profitable_trades': 0,
            'stopped_trades': 0,
            'take_profit_hits': 0,
            'partial_stop_hits': 0,  # ✅ إضافة إحصائية جديدة
            'total_pnl': 0
        }
    
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
    
    def get_price_data(self, symbol, interval='15m', limit=50):
        """الحصول على بيانات السعر"""
        try:
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
        """الحصول على السعر الحالي"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
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
        """مزامنة الصفقات مع Binance"""
        try:
            self.debug_active_positions()
            
            active_positions = self.get_active_positions_from_binance()
            current_managed = set(self.managed_trades.keys())
            binance_symbols = {pos['symbol'] for pos in active_positions}
            
            logger.info(f"🔄 المزامنة: {len(active_positions)} صفقة في Binance, {len(current_managed)} صفقة مدارة")
            
            added_count = 0
            for position in active_positions:
                if position['symbol'] not in current_managed:
                    logger.info(f"🔄 إضافة صفقة جديدة للمراقبة: {position['symbol']}")
                    
                    df = self.get_price_data(position['symbol'])
                    if df is not None and not df.empty:
                        success = self.manage_new_trade(position)
                        if success:
                            logger.info(f"✅ بدء إدارة {position['symbol']} بنجاح")
                            added_count += 1
                        else:
                            logger.error(f"❌ فشل بدء إدارة {position['symbol']}")
                    else:
                        logger.warning(f"⚠️ لا يمكن إدارة {position['symbol']} - بيانات السعر غير متوفرة")
            
            removed_count = 0
            for symbol in list(current_managed):
                if symbol not in binance_symbols:
                    logger.info(f"🔄 إزالة صفقة مغلقة: {symbol}")
                    if symbol in self.managed_trades:
                        del self.managed_trades[symbol]
                        removed_count += 1
            
            logger.info(f"🔄 انتهت المزامنة: أضيف {added_count}، أزيل {removed_count}")
            return len(active_positions)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المزامنة مع Binance: {e}")
            return 0
    
    def manage_new_trade(self, trade_data):
        """بدء إدارة صفقة جديدة مع وقف مزدوج"""
        symbol = trade_data['symbol']
        
        logger.info(f"🔄 بدء إدارة صفقة جديدة: {symbol}")
        
        df = self.get_price_data(symbol)
        if df is None or df.empty:
            logger.error(f"❌ لا يمكن إدارة {symbol} - بيانات السعر غير متوفرة")
            return False
        
        try:
            df = self.stop_loss_manager.calculate_support_resistance(df)
            
            # ✅ حساب وقف الخسارة المزدوج
            stop_loss_levels = self.stop_loss_manager.calculate_dynamic_stop_loss(
                symbol, trade_data['entry_price'], trade_data['direction'], df
            )
            
            take_profit_levels = self.take_profit_manager.calculate_dynamic_take_profit(
                symbol, trade_data['entry_price'], trade_data['direction'], df
            )
            
            if not take_profit_levels:
                logger.warning(f"⚠️ استخدام جني الأرباح الافتراضي لـ {symbol}")
                if trade_data['direction'] == 'LONG':
                    take_profit_levels = {
                        'LEVEL_1': {'price': trade_data['entry_price'] * 1.0025, 'target_percent': 0.25, 'allocation': 0.4, 'quantity': None},
                        'LEVEL_2': {'price': trade_data['entry_price'] * 1.0035, 'target_percent': 0.35, 'allocation': 0.3, 'quantity': None},
                        'LEVEL_3': {'price': trade_data['entry_price'] * 1.0050, 'target_percent': 0.50, 'allocation': 0.3, 'quantity': None}
                    }
                else:
                    take_profit_levels = {
                        'LEVEL_1': {'price': trade_data['entry_price'] * 0.9975, 'target_percent': 0.25, 'allocation': 0.4, 'quantity': None},
                        'LEVEL_2': {'price': trade_data['entry_price'] * 0.9965, 'target_percent': 0.35, 'allocation': 0.3, 'quantity': None},
                        'LEVEL_3': {'price': trade_data['entry_price'] * 0.9950, 'target_percent': 0.50, 'allocation': 0.3, 'quantity': None}
                    }
            
            total_quantity = trade_data['quantity']
            for level, config in take_profit_levels.items():
                config['quantity'] = self.take_profit_manager.calculate_partial_close_quantity(
                    total_quantity, config['allocation']
                )
            
            # حفظ بيانات الإدارة مع الوقف المزدوج
            self.managed_trades[symbol] = {
                **trade_data,
                'dynamic_stop_loss': stop_loss_levels,  # الآن dictionary
                'take_profit_levels': take_profit_levels,
                'closed_levels': [],
                'partial_stop_hit': False,  # تتبع المرحلة الأولى
                'last_update': datetime.now(damascus_tz),
                'status': 'managed',
                'management_start': datetime.now(damascus_tz)
            }
            
            self.performance_stats['total_trades_managed'] += 1
            
            # ✅ إرسال إشعار بدء الإدارة مع تفاصيل الوقف المزدوج
            self.send_management_start_notification(symbol)
            return True
            
        except Exception as e:
            logger.error(f"❌ فشل إدارة الصفقة {symbol}: {e}")
            return False
    
    def check_managed_trades(self):
        """فحص جميع الصفقات المدارة"""
        closed_trades = []
        
        for symbol, trade in list(self.managed_trades.items()):
            try:
                current_price = self.get_current_price(symbol)
                if not current_price:
                    continue
                
                # 1. فحص وقف الخسارة أولاً
                if self.check_stop_loss(symbol, current_price):
                    closed_trades.append(symbol)
                    continue
                
                # 2. فحص جني الأرباح
                self.check_take_profits(symbol, current_price)
                
                # 3. تحديث المستويات الديناميكية كل ساعة
                if (datetime.now(damascus_tz) - trade['last_update']).seconds > 3600:
                    self.update_dynamic_levels(symbol)
                
            except Exception as e:
                logger.error(f"❌ خطأ في فحص الصفقة {symbol}: {e}")
        
        return closed_trades
    
    def check_stop_loss(self, symbol, current_price):
        """فحص وقف الخسارة المزدوج الديناميكي مع تحقق إضافي"""
        if symbol not in self.managed_trades:
            return False
        
        trade = self.managed_trades[symbol]
        stop_levels = trade['dynamic_stop_loss']
    
        # ✅ تحقق إضافي من أن وقف الخسارة ليس قريباً جداً
        entry_price = trade['entry_price']
        min_distance = entry_price * RISK_SETTINGS['min_stop_loss_pct'] * 0.8  # 80% من الحد الأدنى
    
        if trade['direction'] == 'LONG':
            current_distance = entry_price - stop_levels['full_stop_loss']
            if current_distance < min_distance:
                logger.warning(f"⚠️ وقف الخسارة قريب جداً لـ {symbol}. المسافة: {current_distance/entry_price*100:.2f}%")
        else:
            current_distance = stop_levels['full_stop_loss'] - entry_price
            if current_distance < min_distance:
                logger.warning(f"⚠️ وقف الخسارة قريب جداً لـ {symbol}. المسافة: {current_distance/entry_price*100:.2f}%")
    
        should_close_partial = False
        should_close_full = False
    
        if trade['direction'] == 'LONG':
            if current_price <= stop_levels['partial_stop_loss'] and not trade.get('partial_stop_hit'):
                should_close_partial = True
            if current_price <= stop_levels['full_stop_loss']:
                should_close_full = True
        else:  # SHORT
            if current_price >= stop_levels['partial_stop_loss'] and not trade.get('partial_stop_hit'):
                should_close_partial = True
            if current_price >= stop_levels['full_stop_loss']:
                should_close_full = True
    
        # ✅ المرحلة 1: وقف خسارة جزئي
        if should_close_partial:
            logger.info(f"🚨 وقف خسارة جزئي لـ {symbol}: السعر {current_price:.4f}")
        
            close_quantity = trade['quantity'] * RISK_SETTINGS['partial_close_ratio']
            success = self.close_partial_stop_loss(symbol, close_quantity, "وقف خسارة جزئي")
        
            if success:
                trade['partial_stop_hit'] = True
                trade['quantity'] -= close_quantity
                self.performance_stats['partial_stop_hits'] += 1
            
                # إرسال إشعار وقف الخسارة الجزئي
                self.send_partial_stop_loss_notification(trade, current_price, close_quantity, stop_levels)
                return False  # لم يتم إغلاق الصفقة كاملة بعد
    
        # ✅ المرحلة 2: وقف خسارة كامل
        if should_close_full:
            logger.info(f"🚨 وقف خسارة كامل لـ {symbol}: السعر {current_price:.4f}")
        
            success, message = self.close_entire_trade(symbol, "وقف خسارة كامل")
            if success:
                self.performance_stats['stopped_trades'] += 1
            
                pnl_pct = self.calculate_pnl_percentage(trade, current_price)
                self.performance_stats['total_pnl'] += pnl_pct
            
                self.send_trade_closed_notification(trade, current_price, "وقف خسارة كامل", pnl_pct)
                return True
    
        return False
    
    def close_partial_stop_loss(self, symbol, quantity, reason):
        """إغلاق جزئي بسبب وقف الخسارة"""
        try:
            trade = self.managed_trades[symbol]
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if trade['direction'] == 'LONG' else 'BUY',
                type='MARKET',
                quantity=quantity,
                reduceOnly=True
            )
            
            if order:
                logger.info(f"✅ وقف خسارة جزئي لـ {symbol}: {quantity:.6f} - {reason}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في وقف الخسارة الجزئي لـ {symbol}: {e}")
            return False
    
    def check_take_profits(self, symbol, current_price):
        """فحص مستويات جني الأرباح"""
        trade = self.managed_trades[symbol]
        
        for level, config in trade['take_profit_levels'].items():
            if level in trade['closed_levels']:
                continue
            
            should_close = False
            if trade['direction'] == 'LONG' and current_price >= config['price']:
                should_close = True
            elif trade['direction'] == 'SHORT' and current_price <= config['price']:
                should_close = True
            
            if should_close:
                success = self.close_partial_trade(symbol, level, config)
                if success:
                    trade['closed_levels'].append(level)
                    self.performance_stats['take_profit_hits'] += 1
                    
                    # ✅ إرسال إشعار جني الأرباح
                    logger.info(f"📨 جاري إرسال إشعار جني الأرباح {symbol}...")
                    notification_sent = self.send_take_profit_notification(trade, level, current_price)
                    if notification_sent:
                        logger.info(f"✅ تم إرسال إشعار جني الأرباح {symbol}")
                    else:
                        logger.error(f"❌ فشل إرسال إشعار جني الأرباح {symbol}")
                    
                    if len(trade['closed_levels']) == len(trade['take_profit_levels']):
                        self.close_entire_trade(symbol, "تم جني جميع مستويات الربح")
                        self.performance_stats['profitable_trades'] += 1
    
    def close_partial_trade(self, symbol, level, config):
        """إغلاق جزئي للصفقة"""
        try:
            trade = self.managed_trades[symbol]
            quantity = config['quantity']
            
            logger.info(f"🔧 جاري إغلاق جزئي لـ {symbol} - المستوى {level}")
            
            order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if trade['direction'] == 'LONG' else 'BUY',
                type='MARKET',
                quantity=quantity,
                reduceOnly=True
            )
            
            if order:
                logger.info(f"✅ جني رباح جزئي لـ {symbol} - المستوى {level}: {quantity:.6f}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في الجني الجزئي لـ {symbol}: {e}")
            return False
    
    def close_entire_trade(self, symbol, reason):
        """إغلاق كامل للصفقة"""
        try:
            trade = self.managed_trades[symbol]
            
            total_quantity = trade['quantity']
            closed_quantity = sum(
                trade['take_profit_levels'][level]['quantity'] 
                for level in trade['closed_levels'] 
                if level in trade['take_profit_levels']
            )
            remaining_quantity = total_quantity - closed_quantity
            
            if remaining_quantity > 0:
                logger.info(f"🔧 جاري إغلاق كامل لـ {symbol} - الكمية المتبقية: {remaining_quantity}")
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side='SELL' if trade['direction'] == 'LONG' else 'BUY',
                    type='MARKET',
                    quantity=remaining_quantity,
                    reduceOnly=True
                )
                
                if order:
                    if symbol in self.managed_trades:
                        del self.managed_trades[symbol]
                    
                    logger.info(f"✅ إغلاق كامل لـ {symbol}: {reason}")
                    return True, "تم الإغلاق بنجاح"
            
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
        """تحديث مستويات وقف الخسارة وجني الأرباح مع التحقق من الحد الأدنى"""
        if symbol not in self.managed_trades:
            return
    
        trade = self.managed_trades[symbol]
        df = self.get_price_data(symbol)
        if df is None:
            return
    
        # تحديث وقف الخسارة (Trailing Stop)
        df = self.stop_loss_manager.calculate_support_resistance(df)
        new_stop_loss = self.stop_loss_manager.calculate_dynamic_stop_loss(
            symbol, trade['entry_price'], trade['direction'], df
        )
    
        current_price = self.get_current_price(symbol)
        if current_price:
            # ل LONG: نرفع الوقف فقط، ل SHORT: نخفض الوقف فقط
            current_full_stop = trade['dynamic_stop_loss']['full_stop_loss']
            new_full_stop = new_stop_loss['full_stop_loss']
        
            # ✅ التحقق من الحد الأدنى قبل التحديث
            entry_price = trade['entry_price']
            min_distance = entry_price * RISK_SETTINGS['min_stop_loss_pct']
        
            if trade['direction'] == 'LONG':
                current_distance = entry_price - current_full_stop
                new_distance = entry_price - new_full_stop
            
                # تحديث فقط إذا كان الجديد أفضل (أبعد) ولم يتجاوز الحد الأدنى
                if new_full_stop > current_full_stop and new_distance >= min_distance * 0.9:
                    self.managed_trades[symbol]['dynamic_stop_loss'] = new_stop_loss
                    logger.info(f"🔄 تحديث وقف الخسارة لـ {symbol}: جزئي={new_stop_loss['partial_stop_loss']:.4f}, كامل={new_stop_loss['full_stop_loss']:.4f}")
        
            else:  # SHORT
                current_distance = current_full_stop - entry_price
                new_distance = new_full_stop - entry_price
            
                # تحديث فقط إذا كان الجديد أفضل (أقرب) ولم يتجاوز الحد الأدنى
                if new_full_stop < current_full_stop and new_distance >= min_distance * 0.9:
                    self.managed_trades[symbol]['dynamic_stop_loss'] = new_stop_loss
                    logger.info(f"🔄 تحديث وقف الخسارة لـ {symbol}: جزئي={new_stop_loss['partial_stop_loss']:.4f}, كامل={new_stop_loss['full_stop_loss']:.4f}")
    
        self.managed_trades[symbol]['last_update'] = datetime.now(damascus_tz)
    
    def send_management_start_notification(self, symbol):
        """إرسال إشعار بدء الإدارة مع تفاصيل الوقف المزدوج والحد الأدنى"""
        try:
            trade = self.managed_trades[symbol]
            stop_levels = trade['dynamic_stop_loss']
        
            # حساب النسب المئوية
            entry_price = trade['entry_price']
            partial_stop_pct = abs(entry_price - stop_levels['partial_stop_loss']) / entry_price * 100
            full_stop_pct = abs(entry_price - stop_levels['full_stop_loss']) / entry_price * 100
        
            message = (
                f"🔄 <b>بدء إدارة صفقة جديدة</b>\n"
                f"العملة: {symbol}\n"
                f"الاتجاه: {trade['direction']}\n"
                f"سعر الدخول: ${trade['entry_price']:.4f}\n"
                f"الكمية: {trade['quantity']:.6f}\n"
                f"<b>وقف الخسارة المزدوج:</b>\n"
                f"• جزئي (40%): ${stop_levels['partial_stop_loss']:.4f} ({partial_stop_pct:.2f}%)\n"
                f"• كامل (100%): ${stop_levels['full_stop_loss']:.4f} ({full_stop_pct:.2f}%)\n"
                f"<b>الحد الأدنى للوقف:</b> {RISK_SETTINGS['min_stop_loss_pct']*100}%\n"
                f"مستويات جني الأرباح:\n"
            )
        
            for level, config in trade['take_profit_levels'].items():
                tp_pct = abs(entry_price - config['price']) / entry_price * 100
                message += f"• {level}: ${config['price']:.4f} ({tp_pct:.2f}%)\n"
        
            message += f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
        
            return self.notifier.send_message(message)
        
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار بدء الإدارة: {e}")
            return False
    
    def send_partial_stop_loss_notification(self, trade, current_price, closed_quantity, stop_levels):
        """إرسال إشعار وقف خسارة جزئي"""
        try:
            remaining_quantity = trade['quantity']
            entry_price = trade['entry_price']
            
            if trade['direction'] == 'LONG':
                loss_pct = (entry_price - current_price) / entry_price * 100
            else:
                loss_pct = (current_price - entry_price) / entry_price * 100
            
            message = (
                f"🛡️ <b>وقف خسارة جزئي</b>\n"
                f"العملة: {trade['symbol']}\n"
                f"الاتجاه: {trade['direction']}\n"
                f"سعر الدخول: ${entry_price:.4f}\n"
                f"سعر الخروج: ${current_price:.4f}\n"
                f"الخسارة: 🔴 {loss_pct:.2f}%\n"
                f"الكمية المغلقة: {closed_quantity:.6f}\n"
                f"الكمية المتبقية: {remaining_quantity:.6f}\n"
                f"الوقف المتبقي: ${stop_levels['full_stop_loss']:.4f}\n"
                f"السبب: تقليل التعرض للمخاطرة\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار وقف الخسارة الجزئي: {e}")
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
                f"المستويات المتبقية: {len(trade['take_profit_levels']) - len(trade['closed_levels'])}\n"
                f"الوقت: {datetime.now(damascus_tz).strftime('%H:%M:%S')}"
            )
            
            return self.notifier.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال إشعار جني الأرباح: {e}")
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
                f"المستويات المحققة: {len(trade['closed_levels'])}/{len(trade['take_profit_levels'])}\n"
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
                f"🛡️ وقف خسارة جزئي: {self.performance_stats['partial_stop_hits']}\n"
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
            # ✅ اختبار Telegram أولاً
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
                    f"تقنية وقف الخسارة: ديناميكي حسب الدعم/المقاومة + ATR\n"
                    f"تقنية جني الأرباح: 3 مستويات مع تعديل التقلب\n"
                    f"🛡️ نظام الوقف المزدوج: جزئي + كامل\n"
                    f"المراقبة: كل 10 ثواني\n"
                    f"المزامنة: تلقائية مع Binance\n"
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
        """حلقة الإدارة الرئيسية"""
        last_report_time = datetime.now(damascus_tz)
        last_sync_time = datetime.now(damascus_tz)
        
        while True:
            try:
                current_time = datetime.now(damascus_tz)
                
                self.trade_manager.check_managed_trades()
                
                if (current_time - last_sync_time).seconds >= 60:
                    self.trade_manager.monitor_margin_risk()
                    last_sync_time = current_time
                
                if (current_time - last_sync_time).seconds >= 300:
                    self.trade_manager.sync_with_binance_positions()
                    last_sync_time = current_time
                
                if (current_time - last_report_time).seconds >= 21600:
                    self.trade_manager.send_performance_report()
                    last_report_time = current_time
                
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("⏹️ إيقاف البوت يدوياً...")
                break
            except Exception as e:
                logger.error(f"❌ خطأ في حلقة الإدارة: {e}")
                time.sleep(30)

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

@app.route('/api/debug/stop-levels/<symbol>')
def debug_stop_levels(symbol):
    """عرض مستويات وقف الخسارة لعملة محددة"""
    try:
        bot = TradeManagerBot.get_instance()
        
        if symbol in bot.trade_manager.managed_trades:
            trade = bot.trade_manager.managed_trades[symbol]
            current_price = bot.trade_manager.get_current_price(symbol)
            
            return jsonify({
                'success': True,
                'symbol': symbol,
                'entry_price': trade['entry_price'],
                'current_price': current_price,
                'stop_levels': trade['dynamic_stop_loss'],
                'take_profit_levels': trade['take_profit_levels'],
                'timestamp': datetime.now(damascus_tz).isoformat()
            })
        else:
            return jsonify({'success': False, 'message': 'لا توجد صفقة مدارة بهذا الرمز'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def run_flask_app():
    """تشغيل تطبيق Flask"""
    port = int(os.environ.get('PORT', 10001))  # ✅ تغيير إلى PORT ليتوافق مع Render
    app.run(host='0.0.0.0', port=port, debug=False)

def main():
    """الدالة الرئيسية"""
    try:
        bot = TradeManagerBot.get_instance()
        bot.start_management()
        
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        logger.info("🚀 بدء تشغيل مدير الصفقات المتكامل...")
        logger.info(f"🌐 تطبيق Flask يعمل على المنفذ: {os.environ.get('PORT', 10001)}")
        
        # بدء حلقة الإدارة
        bot.management_loop()
                
    except Exception as e:
        logger.error(f"❌ فشل تشغيل البوت: {e}")

if __name__ == "__main__":
    main()
