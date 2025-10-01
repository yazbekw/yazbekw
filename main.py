from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import asyncio
import os
import time
import math
import threading
from datetime import datetime
import logging
import hashlib

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BTC Trading Bot",
    description="Bitcoin technical analysis with Telegram notifications",
    version="2.0.0"
)

# إعدادات التلغرام من متغيرات البيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.recent_messages = {}
        self.message_cooldown = 60

    def send_message(self, message, message_type='info'):
        """إرسال رسالة إلى تلغرام"""
        try:
            if len(message) > 4096:
                message = message[:4090] + "..."

            current_time = time.time()
            message_hash = hashlib.md5(f"{message_type}_{message}".encode()).hexdigest()
            
            # منع تكرار الرسائل
            if message_hash in self.recent_messages:
                if current_time - self.recent_messages[message_hash] < self.message_cooldown:
                    return True

            self.recent_messages[message_hash] = current_time
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id, 
                'text': message, 
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                logger.info("✅ تم إرسال الرسالة إلى التلغرام")
                return True
            else:
                logger.error(f"❌ خطأ في إرسال الرسالة: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال رسالة تلغرام: {e}")
            return False

class BTCAnalyzer:
    """محلل تقني لبيتكوين باستخدام مكتبات خفيفة"""
    
    @staticmethod
    def get_btc_data(days=30):
        """جلب بيانات BTC من CoinGecko"""
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # استخراج الأسعار والحجوم
            prices = [item[1] for item in data['prices']]
            volumes = [item[1] for item in data['total_volumes']]
            
            return {
                'prices': prices,
                'volumes': volumes,
                'current_price': prices[-1] if prices else 0,
                'current_volume': volumes[-1] if volumes else 0
            }
        except Exception as e:
            logger.error(f"Error fetching BTC data: {e}")
            raise
    
    @staticmethod
    def calculate_sma(prices, period):
        """حساب المتوسط المتحرك البسيط"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        return sum(prices[-period:]) / period

    @staticmethod
    def calculate_ema(prices, period):
        """حساب المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for i in range(1, len(prices)):
            ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
        
        return ema

    @staticmethod
    def calculate_rsi(prices, period=14):
        """حساب مؤشر RSI"""
        if len(prices) <= period:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # استخدام آخر فترة
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices):
        """حساب مؤشر MACD"""
        ema_12 = BTCAnalyzer.calculate_ema(prices, 12)
        ema_26 = BTCAnalyzer.calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        # حساب خط الإشارة (EMA 9 لـ MACD)
        macd_history = []
        for i in range(len(prices)):
            ema_12_temp = BTCAnalyzer.calculate_ema(prices[:i+1], 12)
            ema_26_temp = BTCAnalyzer.calculate_ema(prices[:i+1], 26)
            macd_history.append(ema_12_temp - ema_26_temp)
        
        signal_line = BTCAnalyzer.calculate_ema(macd_history, 9) if len(macd_history) >= 9 else macd_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }

    @staticmethod
    def calculate_bollinger_bands(prices, period=20):
        """حساب Bollinger Bands"""
        if len(prices) < period:
            sma = sum(prices) / len(prices) if prices else 0
            std_dev = 0
        else:
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / period
            variance = sum((x - sma) ** 2 for x in recent_prices) / period
            std_dev = math.sqrt(variance)
        
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        current_price = prices[-1] if prices else 0
        
        if upper_band != lower_band:
            position = ((current_price - lower_band) / (upper_band - lower_band)) * 100
        else:
            position = 50
        
        return {
            'sma': sma,
            'upper': upper_band,
            'lower': lower_band,
            'position': position
        }

class TradingBot:
    """بوت التداول الرئيسي"""
    
    def __init__(self):
        self.notifier = None
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        self.analyzer = BTCAnalyzer()
        self.performance_stats = {
            'total_scans': 0,
            'signals_found': 0,
            'trades_executed': 0,
            'last_scan': None,
            'start_time': datetime.now(),
            'last_report': None
        }
        
        self.send_startup_message()

    def calculate_signal_strength(self, value, buy_thresh, sell_thresh):
        """حساب قوة الإشارة من 1 إلى 10"""
        if value <= buy_thresh:
            strength = min(10, int((buy_thresh - value) / buy_thresh * 10) + 1)
            return strength, "شراء"
        elif value >= sell_thresh:
            strength = min(10, int((value - sell_thresh) / (100 - sell_thresh) * 10) + 1)
            return strength, "بيع"
        else:
            return 0, "محايد"

    def analyze_btc_signals(self):
        """تحليل إشارات BTC"""
        try:
            logger.info("🔍 بدء تحليل BTC...")
            data = self.analyzer.get_btc_data(60)
            prices = data['prices']
            
            if len(prices) < 50:
                return {"error": "بيانات غير كافية"}
            
            # حساب المؤشرات
            rsi = self.analyzer.calculate_rsi(prices)
            macd_data = self.analyzer.calculate_macd(prices)
            bb_data = self.analyzer.calculate_bollinger_bands(prices)
            
            # حساب قوة الإشارات
            rsi_strength, rsi_signal = self.calculate_signal_strength(rsi, 30, 70)
            
            macd_signal = "شراء" if macd_data['histogram'] > 0 else "بيع" if macd_data['histogram'] < 0 else "محايد"
            macd_strength = 8 if macd_data['histogram'] > 0 else (8 if macd_data['histogram'] < 0 else 0)
            
            bb_strength, bb_signal = self.calculate_signal_strength(bb_data['position'], 20, 80)
            
            # إشارات إضافية
            sma_20 = self.analyzer.calculate_sma(prices, 20)
            sma_50 = self.analyzer.calculate_sma(prices, 50)
            price_trend = "صاعد" if prices[-1] > sma_20 else "هابط"
            
            # الإشارة العامة
            buy_signals = sum([
                rsi_strength if rsi_signal == "شراء" else 0,
                macd_strength if macd_signal == "شراء" else 0,
                bb_strength if bb_signal == "شراء" else 0
            ])
            
            sell_signals = sum([
                rsi_strength if rsi_signal == "بيع" else 0,
                macd_strength if macd_signal == "بيع" else 0,
                bb_strength if bb_signal == "بيع" else 0
            ])
            
            if buy_signals > sell_signals:
                overall = f"شراء (قوة: {min(10, buy_signals//3)})"
            elif sell_signals > buy_signals:
                overall = f"بيع (قوة: {min(10, sell_signals//3)})"
            else:
                overall = "محايد"
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'price': round(data['current_price'], 2),
                'indicators': {
                    'RSI': {'value': round(rsi, 2), 'strength': rsi_strength, 'signal': rsi_signal},
                    'MACD': {'value': round(macd_data['macd'], 4), 'strength': macd_strength, 'signal': macd_signal},
                    'Bollinger_Bands': {'value': round(bb_data['position'], 2), 'strength': bb_strength, 'signal': bb_signal},
                    'SMA_20': round(sma_20, 2),
                    'SMA_50': round(sma_50, 2),
                    'Price_Trend': price_trend
                },
                'overall_signal': overall,
                'signal_score': buy_signals - sell_signals
            }
            
            self.performance_stats['total_scans'] += 1
            self.performance_stats['signals_found'] += 1 if overall != "محايد" else 0
            self.performance_stats['last_scan'] = datetime.now()
            
            logger.info(f"✅ تم تحليل BTC - الإشارة: {overall}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل الإشارات: {e}")
            return {"error": str(e)}

    async def send_analysis_report(self):
        """إرسال تقرير التحليل إلى التلغرام"""
        try:
            logger.info("📊 إعداد تقرير التحليل...")
            analysis = self.analyze_btc_signals()
            if "error" in analysis:
                error_msg = f"❌ خطأ في التحليل: {analysis['error']}"
                if self.notifier:
                    self.notifier.send_message(error_msg, 'error')
                return
            
            message = f"📊 **تقرير تحليل BTC التلقائي**\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            message += f"💰 السعر: ${analysis['price']:,.2f}\n\n"
            message += "**المؤشرات:**\n"
            
            for name, indicator in analysis['indicators'].items():
                if isinstance(indicator, dict) and 'strength' in indicator:
                    stars = "⭐" * indicator['strength']
                    message += f"• {name}: {indicator['value']} | {indicator['signal']} | قوة: {indicator['strength']}/10 {stars}\n"
                elif name in ['SMA_20', 'SMA_50']:
                    message += f"• {name}: {indicator}\n"
                elif name == 'Price_Trend':
                    message += f"• اتجاه السعر: {indicator}\n"
            
            message += f"\n**🎯 الإشارة العامة: {analysis['overall_signal']}**\n"
            message += f"📈 قوة الإشارة: {analysis['signal_score']} نقطة\n"
            message += "\n⚠️ تحليل فني فقط - ليس نصيحة استثمارية"
            
            if self.notifier:
                success = self.notifier.send_message(message, 'auto_analysis')
                if success:
                    logger.info(f"✅ تم إرسال تقرير التحليل - الإشارة: {analysis['overall_signal']}")
                    self.performance_stats['last_report'] = datetime.now()
                else:
                    logger.error("❌ فشل إرسال تقرير التحليل")
            else:
                logger.warning("⚠️ الإشعارات غير مفعلة - لم يتم إرسال التقرير")
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال تقرير التحليل: {e}")

    async def send_performance_report(self):
        """إرسال تقرير أداء البوت"""
        if not self.notifier:
            logger.warning("⚠️ الإشعارات غير مفعلة - لم يتم إرسال تقرير الأداء")
            return
            
        try:
            current_time = datetime.now()
            uptime = current_time - self.performance_stats['start_time']
            hours = uptime.total_seconds() // 3600
            minutes = (uptime.total_seconds() % 3600) // 60
            
            success_rate = (self.performance_stats['signals_found'] / self.performance_stats['total_scans'] * 100) if self.performance_stats['total_scans'] > 0 else 0
            
            message = f"📈 **تقرير أداء البوت**\n"
            message += f"⏰ وقت التشغيل: {hours:.0f} ساعة {minutes:.0f} دقيقة\n"
            message += f"🔍 إجمالي عمليات المسح: {self.performance_stats['total_scans']}\n"
            message += f"🎯 إشارات تم اكتشافها: {self.performance_stats['signals_found']}\n"
            message += f"📊 معدل النجاح: {success_rate:.1f}%\n"
            message += f"🕒 آخر مسح: {self.performance_stats['last_scan'].strftime('%Y-%m-%d %H:%M:%S') if self.performance_stats['last_scan'] else 'N/A'}\n"
            message += f"📨 آخر تقرير: {self.performance_stats['last_report'].strftime('%Y-%m-%d %H:%M:%S') if self.performance_stats['last_report'] else 'N/A'}\n"
            message += f"⚡ الحالة: 🟢 نشط\n\n"
            message += f"💡 البوت يعمل بشكل طبيعي ويقوم بالفحص كل 30 دقيقة"
            
            success = self.notifier.send_message(message, 'performance_report')
            if success:
                logger.info("✅ تم إرسال تقرير الأداء")
            else:
                logger.error("❌ فشل إرسال تقرير الأداء")
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال تقرير الأداء: {e}")

    def send_startup_message(self):
        """إرسال رسالة بدء التشغيل"""
        if self.notifier:
            message = (
                "🚀 **بدء تشغيل BTC Trading Bot**\n\n"
                "✅ تم تهيئة البوت بنجاح\n"
                "📊 الميزات المتاحة:\n"
                "• تحليل فني متقدم لـ BTC\n"
                "• مؤشرات RSI, MACD, Bollinger Bands\n"
                "• إشعارات تلقائية كل 30 دقيقة\n"
                "• تقارير أداء دورية\n\n"
                f"⏰ وقت البدء: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔔 سيتم إرسال أول تقرير خلال 30 دقيقة"
            )
            self.notifier.send_message(message, 'startup')
            logger.info("✅ تم إرسال رسالة بدء التشغيل")

# تهيئة البوت
bot = TradingBot()

# المهام التلقائية
async def auto_monitoring():
    """المهمة التلقائية للمراقبة"""
    logger.info("🚀 بدء المهام التلقائية...")
    
    # انتظار 5 ثواني عند البدء
    await asyncio.sleep(5)
    
    # إرسال تقرير فوري للاختبار
    await bot.send_analysis_report()
    
    counter = 0
    while True:
        try:
            # كل 30 دقيقة: إرسال تقرير التحليل
            if counter % 6 == 0:  # 30 دقيقة (5 دقائق × 6 = 30)
                logger.info("⏰ وقت إرسال التقرير التلقائي (30 دقيقة)")
                await bot.send_analysis_report()
            
            # كل 6 ساعات: إرسال تقرير الأداء
            if counter % 72 == 0:  # 6 ساعات (5 دقائق × 72 = 360 دقيقة)
                logger.info("📈 وقت إرسال تقرير الأداء (6 ساعات)")
                await bot.send_performance_report()
            
            counter += 1
            logger.info(f"🕒 انتظار 5 دقائق للدورة التالية... (الدورة: {counter})")
            await asyncio.sleep(300)  # انتظار 5 دقائق
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة التلقائية: {e}")
            await asyncio.sleep(60)  # انتظار دقيقة ثم إعادة المحاولة

# Endpoints لـ FastAPI
@app.get("/")
async def root():
    return {
        "message": "مرحباً في BTC Trading Bot",
        "status": "نشط",
        "version": "2.0.0",
        "monitoring": "مفعل كل 30 دقيقة",
        "performance": bot.performance_stats,
        "endpoints": [
            "/analysis", 
            "/health", 
            "/performance", 
            "/test-telegram",
            "/send-report",
            "/send-performance"
        ]
    }

@app.get("/analysis")
async def get_analysis():
    """الحصول على التحليل الحالي"""
    try:
        analysis = bot.analyze_btc_signals()
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في التحليل: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """فحص صحة الخادم"""
    try:
        # اختبار اتصال API
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        api_status = "up" if response.status_code == 200 else "down"
        
        return {
            "status": "healthy",
            "api_status": api_status,
            "timestamp": datetime.now().isoformat(),
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "performance": bot.performance_stats,
            "monitoring_active": True
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/performance")
async def get_performance():
    """الحصول على إحصائيات الأداء"""
    return {
        "performance_stats": bot.performance_stats,
        "current_time": datetime.now().isoformat(),
        "monitoring_status": "نشط"
    }

@app.post("/test-telegram")
async def test_telegram():
    """اختبار إرسال رسالة تلغرام"""
    if not bot.notifier:
        return JSONResponse(
            status_code=400,
            content={"error": "بوت التلغرام غير مهيئ"}
        )
    
    try:
        test_msg = "🧪 **اختبار البوت**\n✅ البوت يعمل بشكل صحيح\n⏰ " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success = bot.notifier.send_message(test_msg, 'test')
        return {"message": "تم إرسال الرسالة التجريبية", "success": success}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في الإرسال: {str(e)}"}
        )

@app.post("/send-report")
async def send_manual_report():
    """إرسال تقرير يدوي"""
    try:
        await bot.send_analysis_report()
        return {"message": "تم إرسال التقرير بنجاح"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في إرسال التقرير: {str(e)}"}
        )

@app.post("/send-performance")
async def send_performance_report():
    """إرسال تقرير أداء يدوي"""
    try:
        await bot.send_performance_report()
        return {"message": "تم إرسال تقرير الأداء بنجاح"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في إرسال تقرير الأداء: {str(e)}"}
        )

# بدء المهام التلقائية عند تشغيل التطبيق
@app.on_event("startup")
async def startup_event():
    """بدء المهام التلقائية عند تشغيل التطبيق"""
    logger.info("🚀 بدء تشغيل المهام التلقائية...")
    asyncio.create_task(auto_monitoring())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
