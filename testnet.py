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
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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

# إعدادات التداول
MAX_LEVERAGE = 20
MAX_POSITION_SIZE = 1000  # USD
MAX_DAILY_LOSS = 200      # USD

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

class TradeTracker:
    """متتبع الصفقات المفتوحة والأداء"""
    
    def __init__(self):
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.performance_stats: Dict[str, float] = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    
    def add_trade(self, symbol: str, side: str, quantity: float, entry_price: float, leverage: int, order_id: str):
        """إضافة صفقة جديدة للمتابعة"""
        trade_id = f"{symbol}_{order_id}"
        self.active_trades[trade_id] = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'leverage': leverage,
            'order_id': order_id,
            'entry_time': datetime.now(),
            'current_pnl': 0.0,
            'pnl_percentage': 0.0,
            'status': 'open'
        }
        logger.info(f"📈 بدء متابعة الصفقة: {trade_id}")
    
    def update_trade_pnl(self, symbol: str, order_id: str, current_price: float, pnl: float, pnl_percentage: float):
        """تحديث PnL للصفقة"""
        trade_id = f"{symbol}_{order_id}"
        if trade_id in self.active_trades:
            self.active_trades[trade_id]['current_pnl'] = pnl
            self.active_trades[trade_id]['pnl_percentage'] = pnl_percentage
            self.active_trades[trade_id]['current_price'] = current_price
            self.active_trades[trade_id]['update_time'] = datetime.now()
    
    def close_trade(self, symbol: str, order_id: str, exit_price: float, pnl: float, pnl_percentage: float):
        """إغلاق صفقة وإضافتها للسجل"""
        trade_id = f"{symbol}_{order_id}"
        if trade_id in self.active_trades:
            trade = self.active_trades[trade_id]
            trade.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'final_pnl': pnl,
                'final_pnl_percentage': pnl_percentage,
                'status': 'closed',
                'duration': (datetime.now() - trade['entry_time']).total_seconds() / 60  # بالدقائق
            })
            
            # نقل للسجل
            self.trade_history.append(trade.copy())
            
            # تحديث الإحصائيات
            self._update_performance_stats(trade)
            
            # إزالة من الصفقات النشطة
            del self.active_trades[trade_id]
            logger.info(f"📉 إغلاق الصفقة: {trade_id} - PnL: {pnl:.4f}")
    
    def _update_performance_stats(self, trade: Dict):
        """تحديث إحصائيات الأداء"""
        self.performance_stats['total_trades'] += 1
        self.performance_stats['total_pnl'] += trade['final_pnl']
        
        if trade['final_pnl'] > 0:
            self.performance_stats['winning_trades'] += 1
            self.performance_stats['largest_win'] = max(self.performance_stats['largest_win'], trade['final_pnl'])
            self.performance_stats['avg_win'] = (
                (self.performance_stats['avg_win'] * (self.performance_stats['winning_trades'] - 1) + trade['final_pnl']) 
                / self.performance_stats['winning_trades']
            )
        else:
            self.performance_stats['losing_trades'] += 1
            self.performance_stats['largest_loss'] = min(self.performance_stats['largest_loss'], trade['final_pnl'])
            self.performance_stats['avg_loss'] = (
                (self.performance_stats['avg_loss'] * (self.performance_stats['losing_trades'] - 1) + trade['final_pnl']) 
                / self.performance_stats['losing_trades']
            )
    
    def get_active_trades(self) -> List[Dict]:
        """الحصول على الصفقات النشطة"""
        return list(self.active_trades.values())
    
    def get_trade_history(self, limit: int = 10) -> List[Dict]:
        """الحصول على سجل الصفقات"""
        return self.trade_history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """الحصول على إحصائيات الأداء"""
        stats = self.performance_stats.copy()
        if stats['total_trades'] > 0:
            stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
        else:
            stats['win_rate'] = 0.0
        return stats

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
            # جرب الإصدار المتوافق مع TESTNET
            return await self._make_request('GET', '/fapi/v1/balance', {}, signed=True)
        except Exception as e:
            logger.warning(f"فشل fapi/v1/balance، جرب fapi/v2: {e}")
            return await self._make_request('GET', '/fapi/v2/balance', {}, signed=True)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """الحصول على معلومات الحساب"""
        return await self._make_request('GET', '/fapi/v2/account', {}, signed=True)
    
    async def get_symbol_price(self, symbol: str) -> float:
        """الحصول على سعر الزوج الحالي"""
        ticker = await self._make_request('GET', '/fapi/v1/ticker/price', {'symbol': symbol})
        return float(ticker['price'])
    
    async def get_exchange_info(self, symbol: str) -> Dict[str, Any]:
        """الحصول على معلومات الزوج"""
        return await self._make_request('GET', '/fapi/v1/exchangeInfo', {'symbol': symbol})
    
    async def get_24h_ticker(self, symbol: str) -> Dict[str, Any]:
        """الحصول على إحصائيات 24 ساعة"""
        return await self._make_request('GET', '/fapi/v1/ticker/24hr', {'symbol': symbol})
    
    async def get_income_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """الحصول على سجل الدخل (الصفقات المغلقة)"""
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        return await self._make_request('GET', '/fapi/v1/income', params, signed=True)

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
            current_price = await self.trader.get_symbol_price(symbol)
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
    """بوت التداول المتقدم للعقود الآجلة مع تتبع الصفقات"""
    
    def __init__(self, telegram_token: str, binance_api_key: str, binance_api_secret: str, testnet: bool = True):
        self.trader = FuturesTrader(binance_api_key, binance_api_secret, testnet)
        self.risk_manager = RiskManager(self.trader)
        self.trade_tracker = TradeTracker()
        self.testnet = testnet
        self.is_running = False
        
        # تهيئة التليجرام
        self.application = Application.builder().token(telegram_token).build()
        self.setup_handlers()
        
        logger.info("🟢 بوت العقود الآجلة المتقدم مع تتبع الصفقات جاهز للتشغيل")
    
    def setup_handlers(self):
        """إعداد معالجات الأوامر"""
        
        # الأوامر الأساسية
        self.application.add_handler(CommandHandler("start", self.handle_start))
        self.application.add_handler(CommandHandler("help", self.handle_help))
        self.application.add_handler(CommandHandler("menu", self.handle_menu))
        self.application.add_handler(CommandHandler("ping", self.handle_ping))
        
        # أوامر التداول
        self.application.add_handler(CommandHandler("long", self.handle_long))
        self.application.add_handler(CommandHandler("short", self.handle_short))
        self.application.add_handler(CommandHandler("close", self.handle_close))
        self.application.add_handler(CommandHandler("close_all", self.handle_close_all))
        self.application.add_handler(CommandHandler("cancel", self.handle_cancel))
        self.application.add_handler(CommandHandler("cancel_all", self.handle_cancel_all))
        
        # أوامر متقدمة
        self.application.add_handler(CommandHandler("limit_long", self.handle_limit_long))
        self.application.add_handler(CommandHandler("limit_short", self.handle_limit_short))
        self.application.add_handler(CommandHandler("stop", self.handle_stop_loss))
        self.application.add_handler(CommandHandler("tp", self.handle_take_profit))
        self.application.add_handler(CommandHandler("leverage", self.handle_leverage))
        
        # أوامر المعلومات والمتابعة
        self.application.add_handler(CommandHandler("positions", self.handle_positions))
        self.application.add_handler(CommandHandler("orders", self.handle_orders))
        self.application.add_handler(CommandHandler("balance", self.handle_balance))
        self.application.add_handler(CommandHandler("price", self.handle_price))
        self.application.add_handler(CommandHandler("info", self.handle_info))
        self.application.add_handler(CommandHandler("stats", self.handle_stats))
        self.application.add_handler(CommandHandler("risk", self.handle_risk))
        
        # أوامر تتبع الصفقات الجديدة
        self.application.add_handler(CommandHandler("trades", self.handle_trades))
        self.application.add_handler(CommandHandler("performance", self.handle_performance))
        self.application.add_handler(CommandHandler("history", self.handle_history))
        
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
    
    async def update_active_trades(self):
        """مهمة دورية لتحديث الصفقات النشطة"""
        logger.info("🔄 بدء مهمة تحديث الصفقات النشطة...")
        while self.is_running:
            try:
                active_trades = self.trade_tracker.get_active_trades()
                if active_trades:
                    logger.debug(f"🔄 تحديث {len(active_trades)} صفقة نشطة")
                    for trade in active_trades:
                        try:
                            # الحصول على السعر الحالي
                            current_price = await self.trader.get_symbol_price(trade['symbol'])
                            
                            # حساب PnL
                            if trade['side'] == 'BUY':  # LONG
                                pnl = (current_price - trade['entry_price']) * trade['quantity']
                            else:  # SHORT
                                pnl = (trade['entry_price'] - current_price) * trade['quantity']
                            
                            pnl_percentage = (pnl / (trade['entry_price'] * trade['quantity'])) * 100 * trade['leverage']
                            
                            # تحديث الصفقة
                            self.trade_tracker.update_trade_pnl(
                                trade['symbol'], 
                                trade['order_id'], 
                                current_price, 
                                pnl, 
                                pnl_percentage
                            )
                            
                        except Exception as e:
                            logger.error(f"خطأ في تحديث الصفقة {trade['symbol']}: {e}")
                
                await asyncio.sleep(10)  # تحديث كل 10 ثواني
                
            except Exception as e:
                logger.error(f"خطأ في مهمة التحديث: {e}")
                await asyncio.sleep(30)
        
        logger.info("🛑 إيقاف مهمة تحديث الصفقات النشطة")
    
    async def check_connection(self) -> bool:
        """التحقق من اتصال البينانس"""
        try:
            await self.trader.get_account_info()
            return True
        except Exception as e:
            logger.error(f"فشل الاتصال ببينانس: {e}")
            return False

    # =========================================================================
    # معالجات الأوامر
    # =========================================================================
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ترحيب"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        welcome_msg = f"""
🤖 *مرحباً بك في بوت العقود الآجلة المتقدم مع تتبع الصفقات* 

🔧 *وضع التشغيل:* {'🟡 TESTNET' if self.testnet else '🟢 MAINNET'}
📊 *الأوامر المتاحة:*

*🟢 أوامر التداول:*
/long symbol quantity [leverage] - شراء طويل
/short symbol quantity [leverage] - بيع قصير  
/close symbol - إغلاق المركز
/close_all - إغلاق جميع المراكز
/cancel symbol - إلغاء أوامر الزوج
/cancel_all - إلغاء جميع الأوامر

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
/info symbol - معلومات الزوج
/stats symbol - إحصائيات 24h
/risk - تقرير المخاطر

*📈 أوامر تتبع الصفقات (جديد):*
/trades - الصفقات النشطة مع PnL مباشر
/performance - إحصائيات أداء الصفقات
/history [limit] - سجل الصفقات المغلقة
/ping - فحص حالة البوت

*💡 أمثلة:*
/long btc 0.01 10x
/short eth 0.5 15x
/limit_long btc 0.01 50000 48000 52000 10x
/stop btc 45000
/trades
/performance
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

*متابعة الصفقات:*
/trades - صفقاتك النشطة
/performance - أدائك
/history - سجل الصفقات

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
            active_trades = len(self.trade_tracker.get_active_trades())
            
            if connection_ok:
                message = (
                    f"🟢 *البوت يعمل بشكل طبيعي*\n"
                    f"• الاتصال ببينانس: ✅\n"
                    f"• الصفقات النشطة: `{active_trades}`\n"
                    f"• وضع التشغيل: `{'TESTNET' if self.testnet else 'MAINNET'}`\n"
                    f"• آخر تحديث: `{datetime.now().strftime('%H:%M:%S')}`\n"
                    f"• حالة المهمة: `{'نشط' if self.is_running else 'متوقف'}`"
                )
            else:
                message = "🔴 فشل الاتصال ببينانس"
            
            await self.send_telegram_message(update, message)
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
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
            
            # الحصول على السعر الحالي
            current_price = await self.trader.get_symbol_price(symbol)
            
            # تغيير الرافعة
            leverage_result = await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'BUY', quantity)
            
            # بدء تتبع الصفقة
            self.trade_tracker.add_trade(symbol, 'BUY', quantity, current_price, leverage, str(order['orderId']))
            
            await self.send_telegram_message(update,
                f"🟢 *تم فتح مركز طويل*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`\n"
                f"📈 *تم بدء متابعة الصفقة*"
            )
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
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
            
            # الحصول على السعر الحالي
            current_price = await self.trader.get_symbol_price(symbol)
            
            # تغيير الرافعة
            await self.trader.change_leverage(symbol, leverage)
            
            # فتح المركز
            order = await self.trader.create_market_order(symbol, 'SELL', quantity)
            
            # بدء تتبع الصفقة
            self.trade_tracker.add_trade(symbol, 'SELL', quantity, current_price, leverage, str(order['orderId']))
            
            await self.send_telegram_message(update,
                f"🔴 *تم فتح مركز قصير*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• السعر: `{current_price:.2f}`\n"
                f"• الرافعة: `{leverage}x`\n"
                f"• النوع: `MARKET`\n"
                f"• المعرف: `{order['orderId']}`\n"
                f"• الحالة: `{order['status']}`\n"
                f"📈 *تم بدء متابعة الصفقة*"
            )
            
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
            
            # بدء تتبع الصفقة
            self.trade_tracker.add_trade(symbol, 'BUY', quantity, entry_price, leverage, str(entry_order['orderId']))
            
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
                f"• معرف الجني: `{tp_order['orderId']}`\n"
                f"📈 *تم بدء متابعة الصفقة*"
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
            
            # بدء تتبع الصفقة
            self.trade_tracker.add_trade(symbol, 'SELL', quantity, entry_price, leverage, str(entry_order['orderId']))
            
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
                f"• معرف الجني: `{tp_order['orderId']}`\n"
                f"📈 *تم بدء متابعة الصفقة*"
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
            current_price = await self.trader.get_symbol_price(symbol)
            
            # إغلاق المركز
            order = await self.trader.create_market_order(symbol, side, quantity)
            
            # إغلاق الصفقة في المتتبع
            position_side = 'BUY' if float(position['positionAmt']) > 0 else 'SELL'
            # البحث عن الصفقة النشطة
            active_trades = self.trade_tracker.get_active_trades()
            for trade in active_trades:
                if trade['symbol'] == symbol and trade['side'] == position_side:
                    self.trade_tracker.close_trade(
                        symbol, 
                        trade['order_id'], 
                        current_price, 
                        unrealized_pnl, 
                        (unrealized_pnl / (entry_price * quantity)) * 100 * trade['leverage']
                    )
                    break
            
            await self.send_telegram_message(update,
                f"🟣 *تم إغلاق المركز*\n"
                f"• الزوج: `{symbol}`\n"
                f"• الكمية: `{quantity}`\n"
                f"• الجانب: `{side}`\n"
                f"• سعر الدخول: `{entry_price}`\n"
                f"• سعر الإغلاق: `{current_price}`\n"
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
                current_price = await self.trader.get_symbol_price(symbol)
                
                try:
                    await self.trader.create_market_order(symbol, side, quantity)
                    
                    # إغلاق الصفقة في المتتبع
                    position_side = 'BUY' if float(position['positionAmt']) > 0 else 'SELL'
                    active_trades = self.trade_tracker.get_active_trades()
                    for trade in active_trades:
                        if trade['symbol'] == symbol and trade['side'] == position_side:
                            entry_price = float(position['entryPrice'])
                            self.trade_tracker.close_trade(
                                symbol, 
                                trade['order_id'], 
                                current_price, 
                                pnl, 
                                (pnl / (entry_price * quantity)) * 100 * trade['leverage']
                            )
                            break
                    
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
            price = await self.trader.get_symbol_price(symbol)
            
            await self.send_telegram_message(update, f"💰 سعر `{symbol}`: `{price}` USDT")
            
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
                f"• الصفقات النشطة: `{len(self.trade_tracker.get_active_trades())}`\n"
                f"• الحد الأقصى للرافعة: `{MAX_LEVERAGE}x`\n"
                f"• الحد الأقصى للمركز: `${MAX_POSITION_SIZE}`\n"
                f"• الحد الأقصى للخسارة: `${MAX_DAILY_LOSS}`\n"
                f"• وضع التشغيل: `{'TESTNET' if self.testnet else 'MAINNET'}`"
            )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    # =========================================================================
    # أوامر تتبع الصفقات الجديدة
    # =========================================================================
    
    async def handle_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض الصفقات النشطة مع PnL مباشر"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            active_trades = self.trade_tracker.get_active_trades()
            
            if not active_trades:
                await self.send_telegram_message(update, "📭 لا توجد صفقات نشطة للمتابعة")
                return
            
            message = "📈 *الصفقات النشطة:*\n\n"
            total_pnl = 0.0
            
            for trade in active_trades:
                pnl = trade.get('current_pnl', 0)
                pnl_percentage = trade.get('pnl_percentage', 0)
                current_price = trade.get('current_price', trade['entry_price'])
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60  # بالدقائق
                
                total_pnl += pnl
                
                pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                side_emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
                
                message += (
                    f"• {side_emoji} {trade['symbol']} {trade['side']}\n"
                    f"  الكمية: `{trade['quantity']}`\n"
                    f"  الدخول: `{trade['entry_price']}`\n"
                    f"  الحالي: `{current_price}`\n"
                    f"  PnL: {pnl_emoji} `{pnl:.4f} USDT` ({pnl_percentage:+.2f}%)\n"
                    f"  الرافعة: `{trade['leverage']}x`\n"
                    f"  المدة: `{duration:.1f} دقيقة`\n"
                    f"  المعرف: `{trade['order_id']}`\n\n"
                )
            
            message += f"*إجمالي PnL النشط:* `{total_pnl:.4f} USDT`"
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض إحصائيات أداء الصفقات"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            stats = self.trade_tracker.get_performance_stats()
            active_trades = self.trade_tracker.get_active_trades()
            
            message = (
                f"📊 *إحصائيات أداء الصفقات*\n\n"
                f"• إجمالي الصفقات: `{stats['total_trades']}`\n"
                f"• الصفقات الناجحة: `{stats['winning_trades']}`\n"
                f"• الصفقات الخاسرة: `{stats['losing_trades']}`\n"
                f"• معدل النجاح: `{stats['win_rate']:.1f}%`\n"
                f"• إجمالي PnL: `{stats['total_pnl']:.4f} USDT`\n"
                f"• متوسط الربح: `{stats['avg_win']:.4f} USDT`\n"
                f"• متوسط الخسارة: `{stats['avg_loss']:.4f} USDT`\n"
                f"• أكبر ربح: `{stats['largest_win']:.4f} USDT`\n"
                f"• أكبر خسارة: `{stats['largest_loss']:.4f} USDT`\n"
                f"• الصفقات النشطة: `{len(active_trades)}`"
            )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض سجل الصفقات المغلقة"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        try:
            limit = int(context.args[0]) if context.args else 10
            history = self.trade_tracker.get_trade_history(limit)
            
            if not history:
                await self.send_telegram_message(update, "📭 لا توجد صفقات مغلقة في السجل")
                return
            
            message = f"📋 *آخر {len(history)} صفقة مغلقة:*\n\n"
            
            for trade in history[-limit:]:
                pnl = trade['final_pnl']
                pnl_percentage = trade['final_pnl_percentage']
                duration = trade['duration']
                
                pnl_emoji = "🟢" if pnl > 0 else "🔴"
                side_emoji = "🟢" if trade['side'] == 'BUY' else "🔴"
                
                message += (
                    f"• {side_emoji} {trade['symbol']} {trade['side']}\n"
                    f"  الدخول: `{trade['entry_price']}`\n"
                    f"  الخروج: `{trade['exit_price']}`\n"
                    f"  PnL: {pnl_emoji} `{pnl:.4f} USDT` ({pnl_percentage:+.2f}%)\n"
                    f"  المدة: `{duration:.1f} دقيقة`\n"
                    f"  الرافعة: `{trade['leverage']}x`\n"
                    f"  التاريخ: `{trade['exit_time'].strftime('%Y-%m-%d %H:%M')}`\n\n"
                )
            
            await self.send_telegram_message(update, message)
            
        except Exception as e:
            await self.send_telegram_message(update, f"❌ خطأ: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة الرسائل العادية"""
        if not await self.is_user_allowed(update.effective_user.id):
            return
        
        await self.send_telegram_message(update, 
            "❓ لم أفهم الأمر. اكتب /help لعرض جميع الأوامر المتاحة."
        )
    
    async def run_async(self):
        """تشغيل البوت بشكل غير متزامن"""
        logger.info("🚀 بدء تشغيل بوت العقود الآجلة المتقدم مع تتبع الصفقات...")
        
        # بدء مهمة تحديث الصفقات
        self.is_running = True
        update_task = asyncio.create_task(self.update_active_trades())
        
        try:
            # تشغيل البوت
            await self.application.run_polling()
        except Exception as e:
            logger.error(f"❌ خطأ في تشغيل البوت: {e}")
        finally:
            # إيقاف مهمة التحديث
            self.is_running = False
            await update_task
    
    def run(self):
        """تشغيل البوت (للتوافق مع الإصدار القديم)"""
        asyncio.run(self.run_async())

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
    main()
