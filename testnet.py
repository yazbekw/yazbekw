import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import httpx
from datetime import datetime
import json
import hmac
import hashlib
import time

# =============================================================================
# الإعدادات الرئيسية - TESTNET
# =============================================================================

# إعدادات TESTNET
TESTNET = True

# مفاتيح TESTNET - احصل عليها من: https://testnet.binancefuture.com/
BINANCE_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', 'your_testnet_api_key_here')
BINANCE_API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', 'your_testnet_api_secret_here')

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'your_telegram_bot_token_here')
ALLOWED_USER_IDS = [int(x) for x in os.getenv('ALLOWED_USER_IDS', '123456789').split(',')]

# إعدادات Binance URLs
BASE_URL = 'https://testnet.binance.vision' if TESTNET else 'https://api.binance.com'
FUTURES_URL = 'https://testnet.binancefuture.com' if TESTNET else 'https://fapi.binance.com'

# إعدادات التداول
MAX_LEVERAGE = 20
MAX_POSITION_SIZE = 1000  # USD

# =============================================================================
# نهاية الإعدادات الرئيسية
# =============================================================================

# إعداد التسجيل
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FuturesTrader:
    """مدير تداول العقود الآجلة باستخدام binance-connector"""
    
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
            
            if signed and params:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._sign_request(params)
            
            async with httpx.AsyncClient() as client:
                if method == 'GET':
                    response = await client.get(url, params=params, headers=headers)
                elif method == 'POST':
                    response = await client.post(url, params=params, headers=headers)
                elif method == 'DELETE':
                    response = await client.delete(url, params=params, headers=headers)
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
        """إنشاء جني أرباح"""
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
        return await self._make_request('GET', '/fapi/v2/balance', {}, signed=True)
    
    async def get_symbol_price(self, symbol: str) -> float:
        """الحصول على سعر الزوج الحالي"""
        ticker = await self._make_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(ticker['price'])

class RiskManager:
    """مدير المخاطر"""
    
    def __init__(self, trader: FuturesTrader):
        self.trader = trader
        self.max_leverage = MAX_LEVERAGE
        self.max_position_size = MAX_POSITION_SIZE
    
    async def validate_leverage(self, leverage: int) -> tuple[bool, str]:
        """التحقق من صحة الرافعة"""
        if leverage > self.max_leverage:
            return False, f"الرافعة {leverage}x تتجاوز الحد الأقصى {self.max_leverage}x"
        return True, "✅ الرافعة مقبولة"
    
    async def validate_position_size(self, symbol: str, quantity: float, leverage: int) -> tuple[bool, str]:
        """التحقق من حجم المركز"""
        try:
            current_price = await self.trader.get_symbol_price(symbol)
            position_size = current_price * quantity * leverage
            
            if position_size > self.max_position_size:
                return False, f"حجم المركز ${position_size:.2f} يتجاوز الحد ${self.max_position_size}"
            
            return True, f"✅ حجم المركز ${position_size:.2f} مقبول"
            
        except Exception as e:
            return False, f"❌ خطأ في حساب حجم المركز: {e}"
    
    def get_daily_pnl(self) -> float:
        """الحصول على أرباح/خسائر اليوم"""
        try:
            account = self.trader.client.futures_account()
            return float(account['totalUnrealizedProfit'])
        except Exception as e:
            logger.error(f"خطأ في جلب PNL اليومي: {e}")
            return 0.0

class AdvancedFuturesBot:
    """بوت التداول المتقدم للعقود الآجلة"""
    
    def __init__(self, telegram_token: str, binance_api_key: str, binance_api_secret: str, testnet: bool = True):
        self.trader = FuturesTrader(binance_api_key, binance_api_secret, testnet)
        self.risk_manager = RiskManager(self.trader)
        self.testnet = testnet
        
        # تهيئة التليجرام
        self.application = Application.builder().token(telegram_token).build()
        self.setup_handlers()
        
        logger.info("🟢 بوت العقود الآجلة جاهز للتشغيل")
    
    def setup_handlers(self):
        """إعداد معالجات الأوامر"""
        
        # الأوامر الأساسية
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        
        # أوامر التداول
        self.application.add_handler(CommandHandler("long", self.handle_long))
        self.application.add_handler(CommandHandler("short", self.handle_short))
        self.application.add_handler(CommandHandler("close", self.handle_close))
        self.application.add_handler(CommandHandler("cancel", self.handle_cancel))
        
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
    
    # =========================================================================
    # معالجات الأوامر
    # =========================================================================
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ترحيب"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        welcome_msg = f"""
🤖 *مرحباً بك في بوت العقود الآجلة* 

🔧 *وضع التشغيل:* {'🟡 TESTNET' if self.testnet else '🟢 MAINNET'}
📊 *الأوامر المتاحة:*

*🟢 أوامر التداول:*
/long symbol quantity [leverage] - شراء طويل
/short symbol quantity [leverage] - بيع قصير  
/close symbol - إغلاق المركز
/cancel symbol - إلغاء جميع الأوامر

*🎯 أوامر الحدود:*
/limit_long symbol quantity entry sl tp [leverage] - شراء حدي
/limit_short symbol quantity entry sl tp [leverage] - بيع حدي

*🛡️ أوامر الوقف:*
/stop symbol price - وقف خسارة
/tp symbol price - جني أرباح
/leverage symbol value - تغيير الرافعة

*📊 أوامر المعلومات:*
/positions - المراكز المفتوحة
/orders [symbol] - الأوامر المعلقة
/balance - رصيد الحساب
/price symbol - سعر الزوج

*💡 أمثلة:*
/long btc 0.01 10x
/limit_long eth 0.5 2500 2400 2600 15x
/stop btc 45000
        """
        await self.send_telegram_message(update, welcome_msg)
    
    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض المساعدة"""
        await self.handle_start(update, context)
    
    async def handle_long(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز طويل"""
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
            leverage_ok, leverage_msg = self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            # تغيير الرافعة
            self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = self.trader.create_market_order(symbol, 'BUY', quantity)
            
            await self.send_telegram_message(update,
                f"🟢 *تم فتح مركز طويل*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_short(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """فتح مركز قصير"""
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
            leverage_ok, leverage_msg = self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            size_ok, size_msg = self.risk_manager.validate_position_size(symbol, quantity, leverage)
            if not size_ok:
                await self.send_telegram_message(update, size_msg)
                return
            
            # تغيير الرافعة
            self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = self.trader.create_market_order(symbol, 'SELL', quantity)
            
            await self.send_telegram_message(update,
                f"🔴 *تم فتح مركز قصير*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            leverage_ok, leverage_msg = self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            self.trader.change_leverage(symbol, leverage)
            
            # أمر الدخول الحدّي
            entry_order = self.trader.create_limit_order(symbol, 'BUY', quantity, entry_price)
            
            # أوامر الوقف
            sl_order = self.trader.create_stop_loss(symbol, 'SELL', stop_loss)
            tp_order = self.trader.create_take_profit(symbol, 'SELL', take_profit)
            
            await self.send_telegram_message(update,
                f"🟡 *تم وضع أمر حدي طويل*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• وقف الخسارة: `{stop_loss}`\n"
                f"• جني الأرباح: `{take_profit}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• معرف الدخول: `{entry_order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            leverage_ok, leverage_msg = self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            self.trader.change_leverage(symbol, leverage)
            
            # أمر الدخول الحدّي
            entry_order = self.trader.create_limit_order(symbol, 'SELL', quantity, entry_price)
            
            # أوامر الوقف
            sl_order = self.trader.create_stop_loss(symbol, 'BUY', stop_loss)
            tp_order = self.trader.create_take_profit(symbol, 'BUY', take_profit)
            
            await self.send_telegram_message(update,
                f"🟡 *تم وضع أمر حدي قصير*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• وقف الخسارة: `{stop_loss}`\n"
                f"• جني الأرباح: `{take_profit}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• معرف الدخول: `{entry_order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            positions = self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            quantity = abs(float(position['positionAmt']))
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # إغلاق المركز
            order = self.trader.create_market_order(symbol, side, quantity)
            
            await self.send_telegram_message(update,
                f"🟣 *تم إغلاق المركز*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• الجانب: `{side}`\n"
                f"• المعرف: `{order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            positions = self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            # إلغاء أوامر الوقف القديمة
            self.trader.cancel_all_orders(symbol)
            
            # تحديد الجانب للوقف
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # وضع وقف جديد
            order = self.trader.create_stop_loss(symbol, side, stop_price)
            
            await self.send_telegram_message(update,
                f"🛡️ *تم تحديث وقف الخسارة*\n"
                f"• الزوج: `{symbol}`\n"
                f"• السعر: `{stop_price}`\n"
                f"• الجانب: `{side}`\n"
                f"• المعرف: `{order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            positions = self.trader.get_position_info(symbol)
            position = next((p for p in positions if float(p['positionAmt']) != 0), None)
            
            if not position:
                await self.send_telegram_message(update, f"❌ لا يوجد مركز مفتوح لـ `{symbol}`")
                return
            
            # تحديد الجانب لجني الأرباح
            side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
            
            # وضع جني الأرباح
            order = self.trader.create_take_profit(symbol, side, tp_price)
            
            await self.send_telegram_message(update,
                f"🎯 *تم تحديث جني الأرباح*\n"
                f"• الزوج: `{symbol}`\n"
                f"• السعر: `{tp_price}`\n"
                f"• الجانب: `{side}`\n"
                f"• المعرف: `{order['orderId']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
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
            leverage_ok, leverage_msg = self.risk_manager.validate_leverage(leverage)
            if not leverage_ok:
                await self.send_telegram_message(update, leverage_msg)
                return
            
            # تغيير الرافعة
            result = self.trader.change_leverage(symbol, leverage)
            
            await self.send_telegram_message(update,
                f"⚡ *تم تغيير الرافعة*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• الحد الأقصى: `{result['maxNotionalValue']}`"
            )
            
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """إلغاء جميع الأوامر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            symbol = context.args[0].upper() + 'USDT' if context.args else None
            
            if symbol:
                result = self.trader.cancel_all_orders(symbol)
                await self.send_telegram_message(update, f"🗑️ تم إلغاء جميع أوامر `{symbol}`")
            else:
                await self.send_telegram_message(update, "❌ usage: /cancel symbol")
                
        except BinanceAPIException as e:
            await self.send_telegram_message(update, f"❌ خطأ Binance: {e.message}")
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض المراكز المفتوحة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            symbol = context.args[0].upper() + 'USDT' if context.args else None
            positions = self.trader.get_position_info(symbol)
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not open_positions:
                await self.send_telegram_message(update, "📭 لا توجد مراكز مفتوحة")
                return
            
            message = "📊 *المراكز المفتوحة:*\n\n"
            for pos in open_positions:
                side = "🟢 LONG" if float(pos['positionAmt']) > 0 else "🔴 SHORT"
                pnl = float(pos['unRealizedProfit'])
                pnl_emoji = "💰" if pnl > 0 else "💸" if pnl < 0 else "⚪"
                
                message += (
                    f"• {pos['symbol']} {side}\n"
                    f"  الكمية: `{abs(float(pos['positionAmt']))}`\n"
                    f"  سعر الدخول: `{pos['entryPrice']}`\n"
                    f"  PnL: {pnl_emoji} `{pnl:.4f} USDT`\n"
                    f"  الرافعة: `{pos['leverage']}x`\n\n"
                )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض الأوامر المعلقة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            symbol = context.args[0].upper() + 'USDT' if context.args else None
            orders = self.trader.get_open_orders(symbol)
            
            if not orders:
                await self.send_telegram_message(update, "📭 لا توجد أوامر معلقة")
                return
            
            message = "📋 *الأوامر المعلقة:*\n\n"
            for order in orders:
                side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"
                message += (
                    f"• {side_emoji} {order['symbol']} - {order['type']}\n"
                    f"  الجانب: `{order['side']}`\n"
                    f"  الكمية: `{order['origQty']}`\n"
                    f"  السعر: `{order.get('price', 'MARKET')}`\n"
                    f"  المعرف: `{order['orderId']}`\n\n"
                )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض رصيد الحساب"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            balance = self.trader.get_account_balance()
            usdt_balance = next((item for item in balance if item['asset'] == 'USDT'), None)
            
            if usdt_balance:
                message = (
                    f"💰 *رصيد العقود الآجلة*\n"
                    f"• الرصيد: `{usdt_balance['balance']} USDT`\n"
                    f"• المتاح: `{usdt_balance['availableBalance']} USDT`\n"
                    f"• الهامش: `{usdt_balance['marginBalance']} USDT`\n"
                    f"• PnL غير المحقق: `{self.risk_manager.get_daily_pnl():.4f} USDT`"
                )
            else:
                message = "❌ لا يمكن جلب بيانات الرصيد"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سعر الزوج"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            if not context.args:
                await self.send_telegram_message(update, "❌ usage: /price symbol")
                return
            
            symbol = context.args[0].upper() + 'USDT'
            price = self.trader.get_symbol_price(symbol)
            
            await self.send_telegram_message(update, f"💰 سعر `{symbol}`: `{price}` USDT")
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {e}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة الرسائل العادية"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        await self.send_telegram_message(update, 
            "❓ لم أفهم الأمر. اكتب /help لعرض جميع الأوامر المتاحة."
        )
    
    def run(self):
        """تشغيل البوت"""
        logger.info("🚀 بدء تشغيل بوت العقود الآجلة...")
        self.application.run_polling()

# =============================================================================
# التشغيل الرئيسي
# =============================================================================

def main():
    """الدالة الرئيسية"""
    
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
        
        bot.run()
        
    except Exception as e:
        logger.error(f"❌ خطأ في تشغيل البوت: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
