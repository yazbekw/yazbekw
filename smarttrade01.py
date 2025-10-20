import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx
import hmac
import hashlib
import time
from datetime import datetime, timedelta
import json
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# =============================================================================
# الإعدادات الرئيسية - TESTNET
# =============================================================================

# إعدادات TESTNET
TESTNET = False

# مفاتيح TESTNET - احصل عليها من: https://testnet.binancefuture.com/
BINANCE_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', 'your_testnet_api_key_here')
BINANCE_API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', 'your_testnet_api_secret_here')

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'your_telegram_bot_token_here')
ALLOWED_USER_IDS = [int(x) for x in os.getenv('ALLOWED_USER_IDS', '123456789').split(',')]

# إعدادات التداول
MAX_LEVERAGE = 20
MAX_POSITION_SIZE = 1000  # USD
MAX_DAILY_LOSS = 200      # USD

# إعدادات الوقف التلقائي
AUTO_STOP_PERCENTAGE = 2.0  # 2% وقف افتراضي
MIN_STOP_DISTANCE = 0.5     # 0.5% أقل مسافة للوقف
MAX_STOP_DISTANCE = 5.0     # 5% أقصى مسافة للوقف

# إعدادات Binance URLs
FUTURES_URL = 'https://testnet.binancefuture.com' if TESTNET else 'https://fapi.binance.com'

# =============================================================================
# نهاية الإعدادات الرئيسية
# =============================================================================

# إعداد التسجيل
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SupportResistanceCalculator:
    """حاسبة مستويات الدعم والمقاومة"""
    
    def __init__(self, trader):
        self.trader = trader
    
    async def get_klines_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[List[Any]]:
        """الحصول على بيانات الشموع"""
        try:
            endpoint = '/fapi/v1/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            return await self.trader._make_request('GET', endpoint, params)
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات الكلاينز: {e}")
            return []
    
    async def calculate_pivot_points(self, symbol: str) -> Dict[str, float]:
        """حساب النقاط المحورية اليومية"""
        try:
            # الحصول على بيانات اليوم السابق
            klines = await self.get_klines_data(symbol, '1d', 2)
            if len(klines) < 2:
                return {}
            
            yesterday = klines[-2]  # اليوم السابق
            high = float(yesterday[2])
            low = float(yesterday[3])
            close = float(yesterday[4])
            
            # حساب النقاط المحورية
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
        except Exception as e:
            logger.error(f"خطأ في حساب النقاط المحورية: {e}")
            return {}
    
    async def calculate_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """حساب مستويات الدعم والمقاومة باستخدام طرق متعددة"""
        try:
            # الحصول على بيانات الأسعار
            klines_1h = await self.get_klines_data(symbol, '1h', 50)
            klines_4h = await self.get_klines_data(symbol, '4h', 50)
            
            if not klines_1h or not klines_4h:
                return {}
            
            # تحويل البيانات
            df_1h = pd.DataFrame(klines_1h, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # تحويل الأعمدة إلى أرقام
            for col in ['high', 'low', 'close']:
                df_1h[col] = pd.to_numeric(df_1h[col])
            
            # حساب النقاط المحورية
            pivot_points = await self.calculate_pivot_points(symbol)
            
            # حساب المتوسطات المتحركة كمستويات دعم/مقاومة ديناميكية
            ma20 = df_1h['close'].tail(20).mean()
            ma50 = df_1h['close'].tail(50).mean()
            
            # تحديد القمم والقيعان الحديثة
            recent_highs = df_1h['high'].tail(24).nlargest(3).tolist()  # آخر 24 ساعة
            recent_lows = df_1h['low'].tail(24).nsmallest(3).tolist()
            
            # الحصول على السعر العادل الحالي
            current_price = await self.trader.get_mark_price(symbol)
            
            # تصنيف المستويات حسب القرب من السعر الحالي
            support_levels = sorted(recent_lows + [ma20, ma50] + 
                                  [pivot_points.get('s1', 0), pivot_points.get('s2', 0), pivot_points.get('s3', 0)])
            resistance_levels = sorted(recent_highs + [ma20, ma50] + 
                                     [pivot_points.get('r1', 0), pivot_points.get('r2', 0), pivot_points.get('r3', 0)])
            
            # ترشيح المستويات ذات المعنى (ليست قريبة جداً من بعضها)
            meaningful_support = []
            meaningful_resistance = []
            
            for level in support_levels:
                if level > 0 and level < current_price:
                    if not meaningful_support or abs(level - meaningful_support[-1]) / current_price > 0.005:
                        meaningful_support.append(level)
            
            for level in resistance_levels:
                if level > 0 and level > current_price:
                    if not meaningful_resistance or abs(level - meaningful_resistance[-1]) / current_price > 0.005:
                        meaningful_resistance.append(level)
            
            # أخذ أقوى 3 مستويات لكل نوع
            strong_support = meaningful_support[-3:] if len(meaningful_support) >= 3 else meaningful_support
            strong_resistance = meaningful_resistance[:3] if len(meaningful_resistance) >= 3 else meaningful_resistance
            
            return {
                'support': strong_support,
                'resistance': strong_resistance,
                'pivot_points': pivot_points,
                'current_price': current_price,
                'ma20': ma20,
                'ma50': ma50
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب الدعم والمقاومة: {e}")
            return {}

class AutoStopManager:
    """مدير الوقف التلقائي لجميع الصفقات"""
    
    def __init__(self, trader, sr_calculator: SupportResistanceCalculator):
        self.trader = trader
        self.sr_calculator = sr_calculator
        self.auto_stop_enabled = True
    
    async def calculate_smart_stop_loss(self, symbol: str, side: str, entry_price: float) -> Tuple[float, str]:
        """حساب وقف الخسارة الذكي بناءً على مستويات الدعم/المقاومة"""
        try:
            # الحصول على مستويات الدعم والمقاومة
            levels = await self.sr_calculator.calculate_support_resistance(symbol)
            
            if not levels or not levels.get('support') or not levels.get('resistance'):
                # إذا فشل الحساب، استخدام النسبة المئوية الافتراضية
                return await self.calculate_percentage_stop(entry_price, side)
            
            # استخدام السعر العادل الحالي مباشرة
            current_price = await self.trader.get_mark_price(symbol)
            reason = ""
            
            if side.upper() == 'BUY':  # صفقة شراء
                # البحث عن أقوى مستوى دعم تحت سعر الدخول
                support_levels = [s for s in levels['support'] if s < entry_price]
                if support_levels:
                    strongest_support = max(support_levels)  # أقوى دعم (أعلى سعر)
                    stop_distance_percent = ((entry_price - strongest_support) / entry_price) * 100
                    
                    # التحقق من أن المسافة معقولة
                    if MIN_STOP_DISTANCE <= stop_distance_percent <= MAX_STOP_DISTANCE:
                        reason = f"أسفل مستوى الدعم القوي ({strongest_support:.2f})"
                        return strongest_support, reason
            
            elif side.upper() == 'SELL':  # صفقة بيع
                # البحث عن أقوى مستوى مقاومة فوق سعر الدخول
                resistance_levels = [r for r in levels['resistance'] if r > entry_price]
                if resistance_levels:
                    strongest_resistance = min(resistance_levels)  # أقوى مقاومة (أقل سعر)
                    stop_distance_percent = ((strongest_resistance - entry_price) / entry_price) * 100
                    
                    # التحقق من أن المسافة معقولة
                    if MIN_STOP_DISTANCE <= stop_distance_percent <= MAX_STOP_DISTANCE:
                        reason = f"فوق مستوى المقاومة القوي ({strongest_resistance:.2f})"
                        return strongest_resistance, reason
            
            # إذا لم نجد مستوى مناسب، استخدام النسبة المئوية
            return await self.calculate_percentage_stop(entry_price, side)
            
        except Exception as e:
            logger.error(f"خطأ في حساب الوقف الذكي: {e}")
            return await self.calculate_percentage_stop(entry_price, side)
    
    async def calculate_percentage_stop(self, entry_price: float, side: str) -> Tuple[float, str]:
        """حساب وقف الخسارة بنسبة مئوية"""
        if side.upper() == 'BUY':
            stop_price = entry_price * (1 - AUTO_STOP_PERCENTAGE / 100)
            reason = f"نسبة {AUTO_STOP_PERCENTAGE}% تحت سعر الدخول"
        else:  # SELL
            stop_price = entry_price * (1 + AUTO_STOP_PERCENTAGE / 100)
            reason = f"نسبة {AUTO_STOP_PERCENTAGE}% فوق سعر الدخول"
        
        return stop_price, reason
    
    async def place_auto_stop_loss(self, symbol: str, side: str, entry_price: float) -> Tuple[bool, str, float]:
        """وضع وقف الخسارة تلقائياً"""
        try:
            if not self.auto_stop_enabled:
                return False, "الوقف التلقائي معطل", 0.0
            
            # حساب سعر الوقف
            stop_price, reason = await self.calculate_smart_stop_loss(symbol, side, entry_price)
            
            # تحديد الجانب المعاكس للوقف
            stop_side = 'SELL' if side.upper() == 'BUY' else 'BUY'
            
            # وضع أمر الوقف
            await self.trader.create_stop_loss(symbol, stop_side, stop_price)
            
            # حساب المسافة كنسبة مئوية
            if side.upper() == 'BUY':
                distance_percent = ((entry_price - stop_price) / entry_price) * 100
            else:
                distance_percent = ((stop_price - entry_price) / entry_price) * 100
            
            return True, reason, distance_percent
            
        except Exception as e:
            logger.error(f"خطأ في وضع الوقف التلقائي: {e}")
            return False, f"خطأ: {str(e)}", 0.0

class FuturesTrader:
    """مدير تداول العقود الآجلة"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = FUTURES_URL
        logger.info(f"🔧 تهيئة Binance Futures {'TESTNET' if testnet else 'MAINNET'}")
    
    def _sign_request(self, params: Dict) -> str:
        """توقيع الطلب"""
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict[str, Any]:
        """تنفيذ طلب HTTP"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {'X-MBX-APIKEY': self.api_key}
            
            if params is None:
                params = {}
            
            if signed:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._sign_request(params)
            
            async with httpx.AsyncClient() as client:
                if method == 'GET':
                    response = await client.get(url, params=params, headers=headers, timeout=30.0)
                elif method == 'POST':
                    response = await client.post(url, params=params, headers=headers, timeout=30.0)
                elif method == 'DELETE':
                    response = await client.delete(url, params=params, headers=headers, timeout=30.0)
                else:
                    raise ValueError(f"Method {method} not supported")
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """تغيير الرافعة المالية"""
        return await self._make_request('POST', '/fapi/v1/leverage', {
            'symbol': symbol,
            'leverage': leverage
        }, signed=True)
    
    async def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """إنشاء أمر سوقي"""
        return await self._make_request('POST', '/fapi/v1/order', {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': quantity
        }, signed=True)
    
    async def create_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """إنشاء أمر حدي"""
        return await self._make_request('POST', '/fapi/v1/order', {
            'symbol': symbol,
            'side': side,
            'type': 'LIMIT',
            'quantity': quantity,
            'price': price,
            'timeInForce': 'GTC'
        }, signed=True)
    
    async def create_stop_loss(self, symbol: str, side: str, stop_price: float) -> Dict[str, Any]:
        """إنشاء وقف خسارة"""
        return await self._make_request('POST', '/fapi/v1/order', {
            'symbol': symbol,
            'side': side,
            'type': 'STOP_MARKET',
            'stopPrice': stop_price,
            'closePosition': 'true'
        }, signed=True)
    
    async def create_take_profit(self, symbol: str, side: str, stop_price: float) -> Dict[str, Any]:
        """إنشاء جني الأرباح"""
        return await self._make_request('POST', '/fapi/v1/order', {
            'symbol': symbol,
            'side': side,
            'type': 'TAKE_PROFIT_MARKET',
            'stopPrice': stop_price,
            'closePosition': 'true'
        }, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """إلغاء جميع الأوامر للزوج"""
        return await self._make_request('DELETE', '/fapi/v1/allOpenOrders', {
            'symbol': symbol
        }, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """إلغاء أمر محدد"""
        return await self._make_request('DELETE', '/fapi/v1/order', {
            'symbol': symbol,
            'orderId': order_id
        }, signed=True)
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """الحصول على الأوامر المفتوحة"""
        params = {'symbol': symbol} if symbol else {}
        return await self._make_request('GET', '/fapi/v1/openOrders', params, signed=True)
    
    async def get_position_info(self, symbol: str = None) -> List[Dict[str, Any]]:
        """الحصول على معلومات المراكز"""
        params = {'symbol': symbol} if symbol else {}
        return await self._make_request('GET', '/fapi/v2/positionRisk', params, signed=True)
    
    async def get_account_balance(self) -> List[Dict[str, Any]]:
        """الحصول على رصيد الحساب"""
        try:
            return await self._make_request('GET', '/fapi/v2/balance', {}, signed=True)
        except Exception as e:
            logger.warning(f"فشل fapi/v2/balance، جرب fapi/v1: {e}")
            return await self._make_request('GET', '/fapi/v1/balance', {}, signed=True)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الحساب"""
        return await self._make_request('GET', '/fapi/v2/account', {}, signed=True)
    
    async def get_symbol_price(self, symbol: str) -> float:
        """الحصول على سعر الزوج الحالي"""
        ticker = await self._make_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(ticker['price'])
    
    async def get_mark_price(self, symbol: str) -> float:
        """الحصول على السعر العادل (Mark Price)"""
        try:
            endpoint = '/fapi/v1/premiumIndex'
            params = {'symbol': symbol}
            data = await self._make_request('GET', endpoint, params)
            return float(data['markPrice'])
        except Exception as e:
            logger.error(f"خطأ في جلب السعر العادل: {e}")
            # Fallback to regular price
            return await self.get_symbol_price(symbol)
    
    async def get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        """الحصول على معلومات الزوج"""
        return await self._make_request('GET', '/fapi/v1/exchangeInfo', {'symbol': symbol})
    
    async def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """الحصول على إحصائيات 24 ساعة"""
        return await self._make_request('GET', '/fapi/v1/ticker/24hr', {'symbol': symbol})

class RiskManager:
    """مدير المخاطر"""
    
    def __init__(self, trader: FuturesTrader):
        self.trader = trader
        self.max_leverage = MAX_LEVERAGE
        self.max_position_size = MAX_POSITION_SIZE
        self.max_daily_loss = MAX_DAILY_LOSS
    
    async def validate_leverage(self, leverage: int) -> Tuple[bool, str]:
        """التحقق من صحة الرافعة"""
        if leverage > self.max_leverage:
            return False, f"الرافعة {leverage}x تتجاوز الحد الأقصى {self.max_leverage}x"
        return True, "✅ الرافعة مقبولة"
    
    async def validate_position_size(self, symbol: str, quantity: float, leverage: int) -> Tuple[bool, str]:
        """التحقق من حجم المركز"""
        try:
            current_price = await self.trader.get_mark_price(symbol)
            position_size = current_price * quantity * leverage
            
            if position_size > self.max_position_size:
                return False, f"حجم المركز ${position_size:.2f} يتجاوز الحد ${self.max_position_size}"
            
            return True, f"✅ حجم المركز ${position_size:.2f} مقبول"
            
        except Exception as e:
            return False, f"❌ خطأ في حساب حجم المركز: {e}"
    
    async def get_daily_pnl(self) -> float:
        """الحصول على أرباح/خسائر اليوم"""
        try:
            account = await self.trader.get_account_info()
            return float(account['totalUnrealizedProfit'])
        except Exception as e:
            logger.error(f"خطأ في جلب PNL اليومي: {e}")
            return 0.0
    
    async def validate_daily_loss(self) -> Tuple[bool, str]:
        """التحقق من الخسارة اليومية"""
        daily_pnl = await self.get_daily_pnl()
        if daily_pnl < -self.max_daily_loss:
            return False, f"وصلت للخسارة اليومية القصوى: ${daily_pnl:.2f}"
        return True, f"✅ الخسارة اليومية: ${daily_pnl:.2f}"

class AdvancedFuturesBot:
    """بوت التداول المتقدم للعقود الآجلة"""
    
    def __init__(self, telegram_token: str, binance_api_key: str, binance_api_secret: str, testnet: bool = True):
        self.trader = FuturesTrader(binance_api_key, binance_api_secret, testnet)
        self.risk_manager = RiskManager(self.trader)
        self.sr_calculator = SupportResistanceCalculator(self.trader)
        self.auto_stop_manager = AutoStopManager(self.trader, self.sr_calculator)
        self.testnet = testnet
        
        # تهيئة التليجرام
        self.application = Application.builder().token(telegram_token).build()
        self.setup_handlers()
        
        logger.info("🟢 بوت العقود الآجلة المتقدم جاهز للتشغيل")
    
    def setup_handlers(self):
        """إعداد معالجات الأوامر"""
        
        # الأوامر الأساسية
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("menu", self.handle_menu))
        self.application.add_handler(CommandHandler("ping", self.handle_ping))
        
        # أوامر التداول الأساسية
        self.application.add_handler(CommandHandler("long", self.handle_long))
        self.application.add_handler(CommandHandler("short", self.handle_short))
        self.application.add_handler(CommandHandler("close", self.handle_close))
        self.application.add_handler(CommandHandler("close_all", self.handle_close_all))
        self.application.add_handler(CommandHandler("cancel", self.handle_cancel))
        self.application.add_handler(CommandHandler("cancel_all", self.handle_cancel_all))
        
        # 🆕 أوامر الدعم والمقاومة والوقف الذكي
        self.application.add_handler(CommandHandler("ls", self.handle_levels_show))
        self.application.add_handler(CommandHandler("lb", self.handle_long_buy_auto_stop))
        self.application.add_handler(CommandHandler("sb", self.handle_short_buy_auto_stop))
        
        # أوامر متقدمة
        self.application.add_handler(CommandHandler("limit_long", self.handle_limit_long))
        self.application.add_handler(CommandHandler("limit_short", self.handle_limit_short))
        self.application.add_handler(CommandHandler("stop", self.handle_stop_loss))
        self.application.add_handler(CommandHandler("tp", self.handle_take_profit))
        self.application.add_handler(CommandHandler("leverage", self.handle_leverage))
        
        # أوامر المعلومات
        self.application.add_handler(CommandHandler("positions", self.handle_positions))
        self.application.add_handler(CommandHandler("orders", self.handle_orders))
        self.application.add_handler(CommandHandler("balance", self.handle_balance))
        self.application.add_handler(CommandHandler("price", self.handle_price))
        self.application.add_handler(CommandHandler("info", self.handle_info))
        self.application.add_handler(CommandHandler("stats", self.handle_stats))
        self.application.add_handler(CommandHandler("risk", self.handle_risk))
        
        # 🆕 أوامر مخصصة لكل عملة - إغلاق
        self.application.add_handler(CommandHandler("cb", self.handle_close_bnb))
        self.application.add_handler(CommandHandler("ce", self.handle_close_eth))
        self.application.add_handler(CommandHandler("cx", self.handle_close_btc))
        self.application.add_handler(CommandHandler("cs", self.handle_close_sol))  # سولانا جديد
        
        # 🆕 أوامر مخصصة لكل عملة - شراء
        self.application.add_handler(CommandHandler("bb", self.handle_buy_bnb))
        self.application.add_handler(CommandHandler("be", self.handle_buy_eth))
        self.application.add_handler(CommandHandler("bx", self.handle_buy_btc))
        self.application.add_handler(CommandHandler("bs", self.handle_buy_sol))  # سولانا جديد
        
        # 🆕 أوامر مخصصة لكل عملة - بيع
        self.application.add_handler(CommandHandler("sb", self.handle_sell_bnb))
        self.application.add_handler(CommandHandler("se", self.handle_sell_eth))
        self.application.add_handler(CommandHandler("sx", self.handle_sell_btc))
        self.application.add_handler(CommandHandler("ss", self.handle_sell_sol))  # سولانا جديد
        
        # 🆕 أوامر مخصصة للسعر
        self.application.add_handler(CommandHandler("pb", self.handle_price_bnb))
        self.application.add_handler(CommandHandler("pe", self.handle_price_eth))
        self.application.add_handler(CommandHandler("px", self.handle_price_btc))
        self.application.add_handler(CommandHandler("ps", self.handle_price_sol))  # سولانا جديد
        
        # معالجة الرسائل العادية
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    async def is_user_allowed(self, user_id: int) -> bool:
        """التحقق من صلاحية المستخدم"""
        return user_id in ALLOWED_USER_IDS
    
    async def send_telegram_message(self, update: Update, message: str):
        """إرسال رسالة مع معالجة الأخطاء"""
        try:
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"خطأ في إرسال الرسالة: {e}")
    
    async def check_connection(self) -> bool:
        """التحقق من اتصال البينانس"""
        try:
            await self.trader.get_account_info()
            return True
        except Exception as e:
            logger.error(f"فشل الاتصال ببينانس: {e}")
            return False

    # =========================================================================
    # 🆕 معالجات الأوامر الجديدة - الدعم والمقاومة والوقف الذكي
    # =========================================================================
    
    async def handle_levels_show(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض مستويات الدعم والمقاومة - ls"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if not context.args:
                await self.send_telegram_message(update, "❌ usage: /ls symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            
            # الحصول على المستويات
            levels = await self.sr_calculator.calculate_support_resistance(symbol)
            
            if not levels:
                await self.send_telegram_message(update, f"❌ لا يمكن حساب المستويات لـ `{symbol}`")
                return
            
            current_price = levels['current_price']
            support_levels = levels.get('support', [])
            resistance_levels = levels.get('resistance', [])
            
            # بناء الرسالة
            message = f"📊 *مستويات الدعم والمقاومة - {symbol}*\n\n"
            message += f"💰 *السعر العادل الحالي:* `{current_price:.2f}`\n\n"
            
            # عرض مستويات الدعم
            if support_levels:
                message += "🟢 *مستويات الدعم:*\n"
                for i, level in enumerate(sorted(support_levels, reverse=True)[:3], 1):
                    distance_percent = ((current_price - level) / current_price) * 100
                    strength = "قوي" if i == 1 else "متوسط" if i == 2 else "ضعيف"
                    message += f"• S{i}: `{level:.2f}` ({distance_percent:.1f}%) - {strength}\n"
            else:
                message += "🟢 *مستويات الدعم:* لا توجد مستويات واضحة\n"
            
            message += "\n"
            
            # عرض مستويات المقاومة
            if resistance_levels:
                message += "🔴 *مستويات المقاومة:*\n"
                for i, level in enumerate(sorted(resistance_levels)[:3], 1):
                    distance_percent = ((level - current_price) / current_price) * 100
                    strength = "قوي" if i == 1 else "متوسط" if i == 2 else "ضعيف"
                    message += f"• R{i}: `{level:.2f}` ({distance_percent:.1f}%) - {strength}\n"
            else:
                message += "🔴 *مستويات المقاومة:* لا توجد مستويات واضحة\n"
            
            message += f"\n💡 *الوقف التلقائي المقترح:*\n"
            message += f"• للشراء: تحت `{support_levels[-1] if support_levels else current_price * 0.98:.2f}`\n"
            message += f"• للبيع: فوق `{resistance_levels[0] if resistance_levels else current_price * 1.02:.2f}`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_long_buy_auto_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شراء ذكي مع وقف تلقائي - lb"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /lb symbol quantity [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            leverage = int(context.args[2].replace('x', '')) if len(context.args) > 2 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = await self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            loss_ok, loss_msg = await self.risk_manager.validate_daily_loss()
            if not loss_ok:
                await self.send_telegram_message(update, loss_msg)
                return
            
            # الحصول على السعر العادل الحالي
            current_price = await self.trader.get_mark_price(symbol)
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'BUY', quantity)
            
            # وضع الوقف التلقائي
            stop_success, stop_reason, stop_distance = await self.auto_stop_manager.place_auto_stop_loss(
                symbol, 'BUY', current_price
            )
            
            # إرسال التأكيد
            message = (
                f"🟢 *تم فتح مركز طويل مع وقف تلقائي*\n\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر العادل: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• المعرف: `{order['orderId']}`\n\n"
            )
            
            if stop_success:
                stop_price = current_price * (1 - stop_distance/100)
                message += (
                    f"🛡️ *الوقف التلقائي:*\n"
                    f"• السعر: `{stop_price:.2f}`\n"
                    f"• المسافة: `{stop_distance:.2f}%`\n"
                    f"• السبب: `{stop_reason}`\n"
                    f"• الحالة: `✅ نشط`"
                )
            else:
                message += f"⚠️ *الوقف التلقائي:* `❌ فشل - {stop_reason}`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_short_buy_auto_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بيع ذكي مع وقف تلقائي - sb"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /sb symbol quantity [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            leverage = int(context.args[2].replace('x', '')) if len(context.args) > 2 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = await self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            loss_ok, loss_msg = await self.risk_manager.validate_daily_loss()
            if not loss_ok:
                await self.send_telegram_message(update, loss_msg)
                return
            
            # الحصول على السعر العادل الحالي
            current_price = await self.trader.get_mark_price(symbol)
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'SELL', quantity)
            
            # وضع الوقف التلقائي
            stop_success, stop_reason, stop_distance = await self.auto_stop_manager.place_auto_stop_loss(
                symbol, 'SELL', current_price
            )
            
            # إرسال التأكيد
            message = (
                f"🔴 *تم فتح مركز قصير مع وقف تلقائي*\n\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر العادل: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• المعرف: `{order['orderId']}`\n\n"
            )
            
            if stop_success:
                stop_price = current_price * (1 + stop_distance/100)
                message += (
                    f"🛡️ *الوقف التلقائي:*\n"
                    f"• السعر: `{stop_price:.2f}`\n"
                    f"• المسافة: `{stop_distance:.2f}%`\n"
                    f"• السبب: `{stop_reason}`\n"
                    f"• الحالة: `✅ نشط`"
                )
            else:
                message += f"⚠️ *الوقف التلقائي:* `❌ فشل - {stop_reason}`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")

    # =========================================================================
    # 🆕 معالجات سولانا الجديدة
    # =========================================================================
    
    async def handle_buy_sol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شراء سولانا بقيمة $5 ورافعة 20x"""
        context.args = ['sol', '5', '20']
        await self.handle_long_buy_auto_stop(update, context)

    async def handle_sell_sol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بيع سولانا بقيمة $5 ورافعة 20x"""
        context.args = ['sol', '5', '20']
        await self.handle_short_buy_auto_stop(update, context)

    async def handle_close_sol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق مركز سولانا"""
        context.args = ['sol']
        await self.handle_close(update, context)

    async def handle_price_sol(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر سولانا"""
        try:
            price = await self.trader.get_mark_price('SOLUSDT')
            await self.send_telegram_message(update, f"💰 سعر `SOL`: `{price}` USDT (السعر العادل)")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ في جلب سعر SOL: {str(e)}")

    # =========================================================================
    # المعالجات الحالية (يتم الحفاظ عليها مع إضافة السعر العادل)
    # =========================================================================
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ترحيب"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        welcome_msg = f"""
🤖 *مرحباً بك في بوت العقود الآجلة المتقدم* 

🔧 *وضع التشغيل:* {'🟡 TESTNET' if self.testnet else '🟢 MAINNET'}
📊 *الأوامر المتاحة:*

*🟢 أوامر التداول الذكية (جديدة):*
/ls symbol - عرض مستويات الدعم والمقاومة
/lb symbol quantity - شراء ذكي مع وقف تلقائي  
/sb symbol quantity - بيع ذكي مع وقف تلقائي

*⚡ أوامر سريعة ($5, 20x) مع وقف تلقائي:*
• `bb` - شراء BNB ذكي    • `sb` - بيع BNB ذكي
• `be` - شراء ETH ذكي    • `se` - بيع ETH ذكي  
• `bx` - شراء BTC ذكي    • `sx` - بيع BTC ذكي
• `bs` - شراء SOL ذكي    • `ss` - بيع SOL ذكي

*💰 أوامر السعر السريعة (باستخدام السعر العادل):*
• `pb` - سعر BNB    • `pe` - سعر ETH
• `px` - سعر BTC    • `ps` - سعر SOL

*🔧 أوامر الإدارة:*
• `cb`, `ce`, `cx`, `cs` - إغلاق المراكز
• `/positions` - المراكز المفتوحة
• `/balance` - رصيد الحساب
• `/risk` - تقرير المخاطر

*💡 أمثلة:*
/ls btc - عرض مستويات BTC
/lb eth 0.5 - شراء ETH ذكي
/sb btc 0.01 - بيع BTC ذكي
bs - شراء سولانا $5, 20x

*🎯 ملاحظة:* جميع الحسابات تستخدم **السعر العادل** لمطابقة بينانس بدقة
        """
        await self.send_telegram_message(update, welcome_msg)

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض المساعدة"""
        await self.handle_start(update, context)
    
    async def handle_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض قائمة الأوامر"""
        menu_msg = """
📋 *القائمة السريعة:*

*تداول سريع:*
/long btc 0.01 - شراء
/short eth 0.5 - بيع
/close btc - إغلاق

*معلومات:*
/positions - مراكزك
/balance - رصيدك  
/price btc - الأسعار

*أوامر سريعة ($5, 20x):*
bb - شراء BNB    | sb - بيع BNB    | cb - إغلاق BNB
be - شراء ETH    | se - بيع ETH    | ce - إغلاق ETH  
bx - شراء BTC    | sx - بيع BTC    | cx - إغلاق BTC
bs - شراء SOL    | ss - بيع SOL    | cs - إغلاق SOL

*أسعار سريعة (السعر العادل):*
pb - سعر BNB    | pe - سعر ETH    | px - سعر BTC    | ps - سعر SOL

*إدارة:*
/stop btc 45000 - وقف
/tp btc 55000 - جني أرباح
/risk - المخاطر
/ping - فحص الحالة
        """
        await self.send_telegram_message(update, menu_msg)
    
    async def handle_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """التحقق من حالة البوت"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            connection_ok = await self.check_connection()
            
            if connection_ok:
                message = (
                    f"🟢 *البوت يعمل بشكل طبيعي*\n"
                    f"• الاتصال ببينانس: ✅\n"
                    f"• وضع التشغيل: `{'TESTNET' if self.testnet else 'MAINNET'}`\n"
                    f"• آخر فحص: `{datetime.now().strftime('%H:%M:%S')}`\n"
                    f"• استخدام السعر العادل: ✅"
                )
            else:
                message = "🔴 فشل الاتصال ببينانس"
            
            await self.send_telegram_message(update, message)
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_long(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز طويل مع وقف تلقائي"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /long symbol quantity [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            leverage = int(context.args[2].replace('x', '')) if len(context.args) > 2 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = await self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            loss_ok, loss_msg = await self.risk_manager.validate_daily_loss()
            if not loss_ok:
                await self.send_telegram_message(update, loss_msg)
                return
            
            # الحصول على السعر العادل الحالي
            current_price = await self.trader.get_mark_price(symbol)
            
            # تغيير الرافعة
            leverage_result = await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'BUY', quantity)
            
            # وضع الوقف التلقائي
            stop_success, stop_reason, stop_distance = await self.auto_stop_manager.place_auto_stop_loss(
                symbol, 'BUY', current_price
            )
            
            # إرسال التأكيد مع معلومات الوقف
            message = (
                f"🟢 *تم فتح مركز طويل*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر العادل: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`\n"
            )
            
            if stop_success:
                stop_price = current_price * (1 - stop_distance/100)
                message += f"🛡️ *الوقف التلقائي:* `{stop_price:.2f}` ({stop_distance:.2f}% - {stop_reason})"
            else:
                message += f"⚠️ *الوقف التلقائي:* `❌ فشل`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_short(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز قصير مع وقف تلقائي"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /short symbol quantity [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            leverage = int(context.args[2].replace('x', '')) if len(context.args) > 2 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = await self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            loss_ok, loss_msg = await self.risk_manager.validate_daily_loss()
            if not loss_ok:
                await self.send_telegram_message(update, loss_msg)
                return
            
            # الحصول على السعر العادل الحالي
            current_price = await self.trader.get_mark_price(symbol)
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'SELL', quantity)
            
            # وضع الوقف التلقائي
            stop_success, stop_reason, stop_distance = await self.auto_stop_manager.place_auto_stop_loss(
                symbol, 'SELL', current_price
            )
            
            # إرسال التأكيد مع معلومات الوقف
            message = (
                f"🔴 *تم فتح مركز قصير*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر العادل: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`\n"
            )
            
            if stop_success:
                stop_price = current_price * (1 + stop_distance/100)
                message += f"🛡️ *الوقف التلقائي:* `{stop_price:.2f}` ({stop_distance:.2f}% - {stop_reason})"
            else:
                message += f"⚠️ *الوقف التلقائي:* `❌ فشل`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_limit_long(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز طويل بحد سعر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 5:
                await self.send_telegram_message(update, "❌ usage: /limit_long symbol quantity entry sl tp [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            entry_price = float(context.args[2])
            stop_loss = float(context.args[3])
            take_profit = float(context.args[4])
            leverage = int(context.args[5].replace('x', '')) if len(context.args) > 5 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # أمر الدخول الحدّي
            entry_order = await self.trader.create_limit_order(symbol, 'BUY', quantity, entry_price)
            
            # أوامر الوقف
            sl_order = await self.trader.create_stop_loss(symbol, 'SELL', stop_loss)
            tp_order = await self.trader.create_take_profit(symbol, 'SELL', take_profit)
            
            await self.send_telegram_message(update,
                f"🟡 *تم وضع أمر حدي طويل*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• وقف الخسارة: `{stop_loss}`\n"
                f"• جني الأرباح: `{take_profit}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• معرف الدخول: `{entry_order['orderId']}`\n"
                f"• معرف الوقف: `{sl_order['orderId']}`\n"
                f"• معرف الجني: `{tp_order['orderId']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_limit_short(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز قصير بحد سعر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 5:
                await self.send_telegram_message(update, "❌ usage: /limit_short symbol quantity entry sl tp [leverage]")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            quantity = float(context.args[1])
            entry_price = float(context.args[2])
            stop_loss = float(context.args[3])
            take_profit = float(context.args[4])
            leverage = int(context.args[5].replace('x', '')) if len(context.args) > 5 else 10
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # أمر الدخول الحدّي
            entry_order = await self.trader.create_limit_order(symbol, 'SELL', quantity, entry_price)
            
            # أوامر الوقف
            sl_order = await self.trader.create_stop_loss(symbol, 'BUY', stop_loss)
            tp_order = await self.trader.create_take_profit(symbol, 'BUY', take_profit)
            
            await self.send_telegram_message(update,
                f"🟡 *تم وضع أمر حدي قصير*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• وقف الخسارة: `{stop_loss}`\n"
                f"• جني الأرباح: `{take_profit}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• معرف الدخول: `{entry_order['orderId']}`\n"
                f"• معرف الوقف: `{sl_order['orderId']}`\n"
                f"• معرف الجني: `{tp_order['orderId']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق المركز"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 1:
                await self.send_telegram_message(update, "❌ usage: /close symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            
            # الحصول على المركز الحالي
            positions = await self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            quantity = abs(float(position['positionAmt']))
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            entry_price = float(position['entryPrice'])
            unrealized_pnl = float(position['unRealizedProfit'])
            
            # إغلاق المركز
            order = await self.trader.create_market_order(symbol, side, quantity)
            
            await self.send_telegram_message(update,
                f"🟣 *تم إغلاق المركز*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• الجانب: `{side}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• PnL: `{unrealized_pnl:.4f} USDT`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق جميع المراكز"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            positions = await self.trader.get_position_info()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not open_positions:
                await self.send_telegram_message(update, "📭 لا توجد مراكز مفتوحة")
                return
            
            closed_count = 0
            total_pnl = 0.0
            
            for position in open_positions:
                symbol = position['symbol']
                quantity = abs(float(position['positionAmt']))
                side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                pnl = float(position['unRealizedProfit'])
                
                try:
                    await self.trader.create_market_order(symbol, side, quantity)
                    closed_count += 1
                    total_pnl += pnl
                    await asyncio.sleep(0.5)  # فواصل بين الأوامر
                except Exception as e:
                    logger.error(f"خطأ في إغلاق {symbol}: {e}")
                    continue
            
            await self.send_telegram_message(update,
                f"🟣 *تم إغلاق جميع المراكز*\n"
                f"• العدد: `{closed_count}`\n"
                f"• PnL الإجمالي: `{total_pnl:.4f} USDT`\n"
                f"• الحالة: `مكتمل`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_stop_loss(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تعديل وقف الخسارة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /stop symbol price")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            stop_price = float(context.args[1])
            
            # الحصول على المركز الحالي
            positions = await self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            # إلغاء أوامر الوقف القديمة
            await self.trader.cancel_all_orders(symbol)
            
            # تحديد الجانب للوقف
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # وضع وقف جديد
            order = await self.trader.create_stop_loss(symbol, side, stop_price)
            
            await self.send_telegram_message(update,
                f"🛡️ *تم تحديث وقف الخسارة*\n"
                f"• الزوج: `{symbol}`\n"
                f"• السعر: `{stop_price}`\n"
                f"• الجانب: `{side}`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_take_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تعديل جني الأرباح"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /tp symbol price")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            tp_price = float(context.args[1])
            
            # الحصول على المركز الحالي
            positions = await self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            # تحديد الجانب لجني الأرباح
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # وضع جني الأرباح
            order = await self.trader.create_take_profit(symbol, side, tp_price)
            
            await self.send_telegram_message(update,
                f"🎯 *تم تحديث جني الأرباح*\n"
                f"• الزوج: `{symbol}`\n"
                f"• السعر: `{tp_price}`\n"
                f"• الجانب: `{side}`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """تغيير الرافعة المالية"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 2:
                await self.send_telegram_message(update, "❌ usage: /leverage symbol value")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            leverage = int(context.args[1])
            
            # التحقق من المخاطر
            leverage_ok, leverage_msg = await self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            result = await self.trader.change_leverage(symbol, leverage)
            
            await self.send_telegram_message(update,
                f"⚡ *تم تغيير الرافعة*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• الحد الأقصى: `{result['maxNotionalValue']}`"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إلغاء أوامر الزوج"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if len(context.args) < 1:
                await self.send_telegram_message(update, "❌ usage: /cancel symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            result = await self.trader.cancel_all_orders(symbol)
            
            await self.send_telegram_message(update, f"🗑️ تم إلغاء جميع أوامر `{symbol}`")
                
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_cancel_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إلغاء جميع الأوامر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            orders = await self.trader.get_open_orders()
            if not orders:
                await self.send_telegram_message(update, "📭 لا توجد أوامر معلقة")
                return
            
            canceled_count = 0
            for order in orders:
                try:
                    await self.trader.cancel_order(order['symbol'], order['orderId'])
                    canceled_count += 1
                    await asyncio.sleep(0.3)  # فواصل بين الإلغاءات
                except Exception as e:
                    logger.error(f"خطأ في إلغاء الأمر {order['orderId']}: {e}")
                    continue
            
            await self.send_telegram_message(update, f"🗑️ تم إلغاء `{canceled_count}` أمر")
                
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض المراكز المفتوحة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            symbol = context.args[0].upper() + 'USDT' if context.args else None
            positions = await self.trader.get_position_info(symbol)
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not open_positions:
                await self.send_telegram_message(update, "📭 لا توجد مراكز مفتوحة")
                return
            
            message = "📊 *المراكز المفتوحة:*\n\n"
            total_pnl = 0.0
            
            for pos in open_positions:
                side = "🟢 LONG" if float(pos['positionAmt']) > 0 else "🔴 SHORT"
                pnl = float(pos['unRealizedProfit'])
                total_pnl += pnl
                pnl_emoji = "💰" if pnl > 0 else "💸" if pnl < 0 else "⚪"
                pnl_percent = (pnl / (float(pos['entryPrice']) * abs(float(pos['positionAmt'])))) * 100
                
                message += (
                    f"• {pos['symbol']} {side}\n"
                    f"  الكمية: `{abs(float(pos['positionAmt']))}`\n"
                    f"  سعر الدخول: `{pos['entryPrice']}`\n"
                    f"  PnL: {pnl_emoji} `{pnl:.4f} USDT` ({pnl_percent:+.2f}%)\n"
                    f"  الرافعة: `{pos['leverage']}x`\n\n"
                )
            
            message += f"*الإجمالي:* `{total_pnl:.4f} USDT`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض الأوامر المعلقة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            symbol = context.args[0].upper() + 'USDT' if context.args else None
            orders = await self.trader.get_open_orders(symbol)
            
            if not orders:
                await self.send_telegram_message(update, "📭 لا توجد أوامر معلقة")
                return
            
            message = "📋 *الأوامر المعلقة:*\n\n"
            for order in orders:
                side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"
                order_type = order['type']
                price = order.get('price', 'MARKET')
                stop_price = order.get('stopPrice', 'N/A')
                
                message += (
                    f"• {side_emoji} {order['symbol']} - {order_type}\n"
                    f"  الجانب: `{order['side']}`\n"
                    f"  الكمية: `{order['origQty']}`\n"
                    f"  السعر: `{price}`\n"
                    f"  وقف: `{stop_price}`\n"
                    f"  المعرف: `{order['orderId']}`\n\n"
                )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض رصيد الحساب"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            balance = await self.trader.get_account_balance()
            usdt_balance = next((item for item in balance if item['asset'] == 'USDT'), None)
            
            if usdt_balance:
                # استخدام البيانات مباشرة من الرصيد
                balance_amount = float(usdt_balance.get('balance', 0))
                available_balance = float(usdt_balance.get('availableBalance', 0))
                margin_balance = float(usdt_balance.get('marginBalance', 0))
                unrealized_pnl = float(usdt_balance.get('unrealizedProfit', 0))
                
                message = (
                    f"💰 *رصيد العقود الآجلة*\n"
                    f"• الرصيد: `{balance_amount:.4f} USDT`\n"
                    f"• المتاح: `{available_balance:.4f} USDT`\n"
                    f"• الهامش: `{margin_balance:.4f} USDT`\n"
                    f"• PnL غير المحقق: `{unrealized_pnl:.4f} USDT`\n"
                    f"• الرصيد الكلي: `{balance_amount + unrealized_pnl:.4f} USDT`"
                )
            else:
                message = "❌ لا يمكن جلب بيانات الرصيد أو لا يوجد رصيد USDT"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ في جلب الرصيد: {str(e)}")
    
    async def handle_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر الزوج"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if not context.args:
                await self.send_telegram_message(update, "❌ usage: /price symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            price = await self.trader.get_mark_price(symbol)
            
            await self.send_telegram_message(update, f"💰 سعر `{symbol}`: `{price}` USDT (السعر العادل)")
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض معلومات الزوج"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if not context.args:
                await self.send_telegram_message(update, "❌ usage: /info symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            info = await self.trader.get_exchange_info(symbol)
            symbol_info = info['symbols'][0]
            
            filters = {f['filterType']: f for f in symbol_info['filters']}
            
            message = (
                f"📈 *معلومات الزوج:* `{symbol}`\n"
                f"• الحالة: `{symbol_info['status']}`\n"
                f"• قاعدة: `{symbol_info['baseAsset']}`\n"
                f"• الاقتباس: `{symbol_info['quoteAsset']}`\n"
                f"• حجم العقد: `{filters['LOT_SIZE']['stepSize']}`\n"
                f"• الحد الأدنى: `{filters['LOT_SIZE']['minQty']}`\n"
                f"• الحد الأقصى: `{filters['LOT_SIZE']['maxQty']}`\n"
                f"• دقة السعر: `{filters['PRICE_FILTER']['tickSize']}`"
            )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض إحصائيات 24 ساعة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if not context.args:
                await self.send_telegram_message(update, "❌ usage: /stats symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            stats = await self.trader.get_24h_ticker(symbol)
            
            price_change = float(stats['priceChange'])
            price_change_percent = float(stats['priceChangePercent'])
            high_price = float(stats['highPrice'])
            low_price = float(stats['lowPrice'])
            volume = float(stats['volume'])
            quote_volume = float(stats['quoteVolume'])
            
            message = (
                f"📊 *إحصائيات 24h:* `{symbol}`\n"
                f"• التغير: `{price_change:+.4f}` ({price_change_percent:+.2f}%)\n"
                f"• الأعلى: `{high_price}`\n"
                f"• الأدنى: `{low_price}`\n"
                f"• الحجم: `{volume:.2f} {symbol.replace('USDT', '')}`\n"
                f"• حجم الاقتباس: `{quote_volume:.2f} USDT`"
            )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض تقرير المخاطر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            daily_pnl = await self.risk_manager.get_daily_pnl()
            positions = await self.trader.get_position_info()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            message = (
                f"🛡️ *تقرير المخاطر*\n"
                f"• PnL اليوم: `{daily_pnl:.4f} USDT`\n"
                f"• المراكز المفتوحة: `{len(open_positions)}`\n"
                f"• الحد الأقصى للرافعة: `{MAX_LEVERAGE}x`\n"
                f"• الحد الأقصى للمركز: `${MAX_POSITION_SIZE}`\n"
                f"• الحد الأقصى للخسارة: `${MAX_DAILY_LOSS}`\n"
                f"• وضع التشغيل: `{'TESTNET' if self.testnet else 'MAINNET'}`\n"
                f"• استخدام السعر العادل: ✅"
            )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    # =========================================================================
    # 🆕 أوامر مخصصة لكل عملة - إغلاق
    # =========================================================================
    
    async def handle_close_bnb(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق مركز BNB"""
        context.args = ['bnb']
        await self.handle_close(update, context)

    async def handle_close_eth(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق مركز ETH"""
        context.args = ['eth']
        await self.handle_close(update, context)

    async def handle_close_btc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إغلاق مركز BTC"""
        context.args = ['btc']
        await self.handle_close(update, context)

    # =========================================================================
    # 🆕 أوامر مخصصة لكل عملة - شراء
    # =========================================================================
    
    async def handle_buy_bnb(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شراء BNB بقيمة $5 ورافعة 20x"""
        context.args = ['bnb', '5', '20']
        await self.handle_long(update, context)

    async def handle_buy_eth(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شراء ETH بقيمة $5 ورافعة 20x"""
        context.args = ['eth', '5', '20']
        await self.handle_long(update, context)

    async def handle_buy_btc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """شراء BTC بقيمة $5 ورافعة 20x"""
        context.args = ['btc', '5', '20']
        await self.handle_long(update, context)

    # =========================================================================
    # 🆕 أوامر مخصصة لكل عملة - بيع
    # =========================================================================
    
    async def handle_sell_bnb(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بيع BNB بقيمة $5 ورافعة 20x"""
        context.args = ['bnb', '5', '20']
        await self.handle_short(update, context)

    async def handle_sell_eth(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بيع ETH بقيمة $5 ورافعة 20x"""
        context.args = ['eth', '5', '20']
        await self.handle_short(update, context)

    async def handle_sell_btc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بيع BTC بقيمة $5 ورافعة 20x"""
        context.args = ['btc', '5', '20']
        await self.handle_short(update, context)

    # =========================================================================
    # 🆕 أوامر مخصصة للسعر
    # =========================================================================
    
    async def handle_price_bnb(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر BNB"""
        try:
            price = await self.trader.get_mark_price('BNBUSDT')
            await self.send_telegram_message(update, f"💰 سعر `BNB`: `{price}` USDT (السعر العادل)")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ في جلب سعر BNB: {str(e)}")

    async def handle_price_eth(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر ETH"""
        try:
            price = await self.trader.get_mark_price('ETHUSDT')
            await self.send_telegram_message(update, f"💰 سعر `ETH`: `{price}` USDT (السعر العادل)")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ في جلب سعر ETH: {str(e)}")

    async def handle_price_btc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر BTC"""
        try:
            price = await self.trader.get_mark_price('BTCUSDT')
            await self.send_telegram_message(update, f"💰 سعر `BTC`: `{price}` USDT (السعر العادل)")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ في جلب سعر BTC: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة الرسائل العادية"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        await self.send_telegram_message(update, 
            "❓ لم أفهم الأمر. اكتب /help لعرض جميع الأوامر المتاحة."
        )
    
    def run(self):
        """تشغيل البوت"""
        logger.info("🚀 بدء تشغيل بوت العقود الآجلة المتقدم...")
        self.application.run_polling()

# =============================================================================
# التشغيل الرئيسي
# =============================================================================

# =============================================================================
# إعدادات الخادم للتشغيل على Render
# =============================================================================
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Bot is running')
    
    def log_message(self, format, *args):
        return  # تعطيل التسجيل

def start_health_check_server():
    """تشغيل خادم للتحقق من صحة التطبيق"""
    port = int(os.getenv('PORT', 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f"🔄 بدء خادم التحقق على المنفذ {port}")
    server.serve_forever()

def main():
    """الدالة الرئيسية"""
    
    # 🔒 منع التشغيل المتعدد
    try:
        import socket
        lock_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        lock_socket.bind(('localhost', 65432))
        logger.info("🔒 قفل التشغيل المفرد مفعل")
    except socket.error:
        logger.error("❌ البوت يعمل بالفعل! أوقف النسخة الأخرى أولاً")
        print("❌ البوت يعمل بالفعل! أوقف النسخة الأخرى أولاً")
        return
    
    # بدء خادم التحقق في thread منفصل
    health_thread = threading.Thread(target=start_health_check_server, daemon=True)
    health_thread.start()
    
    # التحقق من وجود المفاتيح
    if BINANCE_API_KEY == 'your_testnet_api_key_here':
        print("❌ يرجى تعيين مفاتيح TESTNET في الإعدادات")
        return
    
    if TELEGRAM_TOKEN == 'your_telegram_bot_token_here':
        print("❌ يرجى تعيين توكن التليجرام في الإعدادات")
        return
    
    try:
        # إنشاء وتشغيل البوت
        bot = AdvancedFuturesBot(
            telegram_token=TELEGRAM_TOKEN,
            binance_api_key=BINANCE_API_KEY,
            binance_api_secret=BINANCE_API_SECRET,
            testnet=TESTNET
        )
        
        # 🔥 استخدام polling مع إعدادات خاصة
        bot.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # تجاهل الرسائل القديمة
            close_loop=False
        )
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")

if __name__ == "__main__":
    main()
