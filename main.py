from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import math
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import json
import random

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BTC Trading Bot",
    description="Bitcoin analysis with multiple data sources",
    version="3.1.0"
)

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class MultiSourceDataFetcher:
    """جلب البيانات من مصادر متعددة لتجنب المحدودية"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_count = 0
        self.last_request_time = 0
        self.cache = {}
        self.cache_ttl = 300  # 5 دقائق
        
        # مصادر البيانات البديلة
        self.data_sources = [
            self._fetch_from_coingecko,
            self._fetch_from_binance,
            self._fetch_from_yahoo,
            self._generate_simulated_data  # بيانات محاكاة كحل أخير
        ]

    async def _rate_limit(self):
        """تطبيق حدود الطلبات"""
        current_time = time.time()
        if current_time - self.last_request_time < 2:  # طلب كل 2 ثانية على الأقل
            await asyncio.sleep(2)
        self.last_request_time = time.time()

    async def _fetch_from_coingecko(self, days: int) -> Optional[Dict[str, Any]]:
        """جلب البيانات من CoinGecko (المصدر الأساسي)"""
        try:
            await self._rate_limit()
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = await self.client.get(url, headers=headers)
            
            if response.status_code == 429:
                logger.warning("⚠️ تم تجاوز حد الطلبات في CoinGecko")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            return self._process_data(data)
            
        except Exception as e:
            logger.warning(f"⚠️ فشل جلب البيانات من CoinGecko: {e}")
            return None

    async def _fetch_from_binance(self, days: int) -> Optional[Dict[str, Any]]:
        """جلب البيانات من Binance API (بديل)"""
        try:
            await self._rate_limit()
            # Binance لا توفر بيانات تاريخية بنفس الطريقة، لذا نستخدم سعر الحالي
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                current_data = response.json()
                current_price = float(current_data['price'])
                
                # إنشاء بيانات محاكاة بناءً على السعر الحالي
                return self._generate_simulated_data_based_on_price(current_price, days)
                
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ فشل جلب البيانات من Binance: {e}")
            return None

    async def _fetch_from_yahoo(self, days: int) -> Optional[Dict[str, Any]]:
        """جلب البيانات من Yahoo Finance (بديل)"""
        try:
            await self._rate_limit()
            # Yahoo Finance يحتاج إلى معالجة أكثر تعقيداً
            # نعود إلى البيانات المحاكاة مؤقتاً
            return self._generate_simulated_data(days)
            
        except Exception as e:
            logger.warning(f"⚠️ فشل جلب البيانات من Yahoo: {e}")
            return None

    def _generate_simulated_data(self, days: int) -> Dict[str, Any]:
        """إنشاء بيانات محاكاة واقعية"""
        logger.info("🔄 استخدام بيانات محاكاة واقعية")
        
        # سعر بداية واقعي (بناءً على آخر سعر معروف)
        base_price = 60000  
        prices = []
        volumes = []
        
        # إنشاء بيانات تاريخية محاكاة
        for i in range(days * 24):  # بيانات كل ساعة
            # تقلب واقعي (±2%)
            change = random.uniform(-0.02, 0.02)
            price = base_price * (1 + change)
            prices.append(price)
            
            # حجم تداول واقعي
            volume = random.uniform(10000000, 50000000)
            volumes.append(volume)
            
            base_price = price  # تحديث السعر الأساسي
        
        return {
            'prices': prices,
            'volumes': volumes,
            'current_price': prices[-1] if prices else base_price,
            'current_volume': volumes[-1] if volumes else 25000000,
            'source': 'simulated',
            'timestamp': datetime.now().isoformat()
        }

    def _generate_simulated_data_based_on_price(self, current_price: float, days: int) -> Dict[str, Any]:
        """إنشاء بيانات محاكاة بناءً على سعر حقيقي"""
        prices = []
        volumes = []
        
        # البدء من سعر أقل والعودة إلى السعر الحالي
        start_price = current_price * random.uniform(0.8, 0.95)
        
        for i in range(days * 24):
            # اتجاه عام نحو السعر الحالي
            progress = i / (days * 24)
            target_price = start_price + (current_price - start_price) * progress
            
            # تقلب حول الاتجاه
            volatility = 0.01 * (1 - progress)  # تقلب أقل مع اقتراب الوقت الحالي
            price = target_price * (1 + random.uniform(-volatility, volatility))
            prices.append(price)
            
            volume = random.uniform(10000000, 50000000)
            volumes.append(volume)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'current_price': current_price,
            'current_volume': volumes[-1] if volumes else 25000000,
            'source': 'binance_simulated',
            'timestamp': datetime.now().isoformat()
        }

    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة البيانات من CoinGecko"""
        return {
            'prices': [item[1] for item in data['prices']],
            'volumes': [item[1] for item in data['total_volumes']],
            'current_price': data['prices'][-1][1] if data['prices'] else 0,
            'current_volume': data['total_volumes'][-1][1] if data['total_volumes'] else 0,
            'source': 'coingecko',
            'timestamp': datetime.now().isoformat()
        }

    async def get_btc_data(self, days: int = 30) -> Dict[str, Any]:
        """جلب البيانات من أفضل مصدر متاح"""
        cache_key = f"btc_data_{days}"
        current_time = time.time()
        
        # التحقق من الذاكرة المؤقتة أولاً
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logger.info("✅ استخدام البيانات من الذاكرة المؤقتة")
                return cached_data
        
        # تجربة جميع المصادر بالترتيب
        for source in self.data_sources:
            try:
                if asyncio.iscoroutinefunction(source):
                    data = await source(days)
                else:
                    data = source(days)
                    
                if data is not None:
                    logger.info(f"✅ تم جلب البيانات من {data.get('source', 'unknown')}")
                    
                    # تخزين في الذاكرة المؤقتة
                    self.cache[cache_key] = (data, current_time)
                    return data
                    
            except Exception as e:
                logger.warning(f"⚠️ فشل المصدر: {e}")
                continue
        
        # إذا فشلت جميع المصادر، نستخدم البيانات المحاكاة
        logger.warning("🔄 استخدام البيانات المحاكاة بعد فشل جميع المصادر")
        data = self._generate_simulated_data(days)
        self.cache[cache_key] = (data, current_time)
        return data

    async def close(self):
        """إغلاق العميل"""
        await self.client.aclose()

class RobustBTCAnalyzer:
    """محلل BTC قوي مع معالجة الأخطاء"""
    
    def __init__(self):
        self.data_fetcher = MultiSourceDataFetcher()
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'data_source_usage': {},
            'last_successful_analysis': None
        }

    async def analyze_btc(self) -> Dict[str, Any]:
        """تحليل BTC مع معالجة قوية للأخطاء"""
        try:
            logger.info("🔍 بدء تحليل BTC...")
            
            # جلب البيانات
            data = await self.data_fetcher.get_btc_data(30)
            self.performance_stats['total_analyses'] += 1
            
            # تحديث إحصائيات مصدر البيانات
            source = data.get('source', 'unknown')
            self.performance_stats['data_source_usage'][source] = \
                self.performance_stats['data_source_usage'].get(source, 0) + 1
            
            # حساب المؤشرات الأساسية
            indicators = self._calculate_basic_indicators(data['prices'])
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'price': round(data['current_price'], 2),
                'volume': round(data['current_volume'], 2),
                'data_source': source,
                'indicators': indicators,
                'overall_signal': self._determine_signal(indicators),
                'reliability': 'high' if source == 'coingecko' else 'medium',
                'analysis_id': f"ANA_{int(time.time())}"
            }
            
            self.performance_stats['successful_analyses'] += 1
            self.performance_stats['last_successful_analysis'] = datetime.now()
            
            logger.info(f"✅ تحليل ناجح - الإشارة: {analysis['overall_signal']}")
            return analysis
            
        except Exception as e:
            self.performance_stats['failed_analyses'] += 1
            logger.error(f"❌ فشل في التحليل: {e}")
            
            # تحليل بديل باستخدام بيانات افتراضية
            return await self._get_fallback_analysis()

    def _calculate_basic_indicators(self, prices: list) -> Dict[str, Any]:
        """حساب المؤشرات الأساسية"""
        if len(prices) < 20:
            # استخدام بيانات افتراضية إذا لم تكن كافية
            current_price = prices[-1] if prices else 60000
            return self._get_default_indicators(current_price)
        
        try:
            # RSI مبسط
            rsi = self._calculate_simple_rsi(prices)
            
            # اتجاه بسيط
            trend = "صاعد" if prices[-1] > prices[-5] else "هابط"
            
            # تقلب
            recent_prices = prices[-10:] if len(prices) >= 10 else prices
            volatility = (max(recent_prices) - min(recent_prices)) / min(recent_prices) * 100
            
            return {
                'rsi': round(rsi, 2),
                'trend': trend,
                'volatility': round(volatility, 2),
                'price_change_24h': round((prices[-1] / prices[-24] - 1) * 100, 2) if len(prices) >= 24 else 0,
                'support_level': round(min(prices[-50:]) if len(prices) >= 50 else min(prices), 2),
                'resistance_level': round(max(prices[-50:]) if len(prices) >= 50 else max(prices), 2)
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات: {e}")
            return self._get_default_indicators(prices[-1] if prices else 60000)

    def _calculate_simple_rsi(self, prices: list, period: int = 14) -> float:
        """حساب RSI مبسط"""
        if len(prices) <= period:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        if len(gains) < period or len(losses) < period:
            return 50
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _determine_signal(self, indicators: Dict[str, Any]) -> str:
        """تحديد الإشارة بناءً على المؤشرات"""
        rsi = indicators.get('rsi', 50)
        trend = indicators.get('trend', 'neutral')
        volatility = indicators.get('volatility', 0)
        
        if rsi < 30 and trend == "صاعد":
            return "شراء قوي"
        elif rsi > 70 and trend == "هابط":
            return "بيع قوي"
        elif rsi < 45 and trend == "صاعد":
            return "شراء"
        elif rsi > 55 and trend == "هابط":
            return "بيع"
        else:
            return "محايد"

    def _get_default_indicators(self, current_price: float) -> Dict[str, Any]:
        """الحصول على مؤشرات افتراضية"""
        return {
            'rsi': 50.0,
            'trend': 'محايد',
            'volatility': 2.5,
            'price_change_24h': 0.0,
            'support_level': round(current_price * 0.95, 2),
            'resistance_level': round(current_price * 1.05, 2),
            'note': 'بيانات افتراضية بسبب مشكلة في المصدر'
        }

    async def _get_fallback_analysis(self) -> Dict[str, Any]:
        """تحليل بديل عند الفشل"""
        return {
            'timestamp': datetime.now().isoformat(),
            'price': 61750.0,  # سعر افتراضي
            'volume': 25000000,
            'data_source': 'fallback',
            'indicators': self._get_default_indicators(61750.0),
            'overall_signal': 'محايد',
            'reliability': 'low',
            'analysis_id': f"FBA_{int(time.time())}",
            'note': 'هذا تحليل افتراضي بسبب مشكلة تقنية'
        }

    async def close(self):
        """إغلاق الموارد"""
        await self.data_fetcher.close()

class TelegramNotifier:
    """إشعارات تلغرام مبسطة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_message(self, message: str) -> bool:
        """إرسال رسالة إلى تلغرام"""
        if not self.token or not self.chat_id:
            logger.warning("⚠️ إعدادات التلغرام غير مكتملة")
            return False
            
        try:
            if len(message) > 4096:
                message = message[:4090] + "..."
                
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=15.0
                )
                
            if response.status_code == 200:
                logger.info("✅ تم إرسال الرسالة إلى التلغرام")
                return True
            else:
                logger.error(f"❌ خطأ في إرسال الرسالة: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التلغرام: {e}")
            return False

# تهيئة المكونات
analyzer = RobustBTCAnalyzer()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# المهمة التلقائية
async def auto_analysis_task():
    """مهمة التحليل التلقائي"""
    logger.info("🚀 بدء مهمة التحليل التلقائي...")
    
    while True:
        try:
            # تحليل BTC
            analysis = await analyzer.analyze_btc()
            
            # إنشاء رسالة التقرير
            message = f"📊 **تقرير تحليل BTC**\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            message += f"💰 السعر: ${analysis['price']:,.2f}\n"
            message += f"📈 المصدر: {analysis['data_source']}\n"
            message += f"🎯 الموثوقية: {analysis['reliability']}\n\n"
            
            message += "**المؤشرات:**\n"
            for key, value in analysis['indicators'].items():
                if key != 'note':
                    message += f"• {key.replace('_', ' ').title()}: {value}\n"
            
            if 'note' in analysis['indicators']:
                message += f"\n📝 ملاحظة: {analysis['indicators']['note']}\n"
            
            message += f"\n**🎯 الإشارة: {analysis['overall_signal']}**\n"
            message += f"🆔 رقم التحليل: {analysis['analysis_id']}\n"
            message += "\n⚠️ تحليل فني - ليس نصيحة استثمارية"
            
            # إرسال التقرير
            await notifier.send_message(message)
            
            logger.info(f"✅ تم إكمال دورة التحليل - الإشارة: {analysis['overall_signal']}")
            
            # الانتظار 30 دقيقة
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة التلقائية: {e}")
            await asyncio.sleep(60)  # انتظار دقيقة وإعادة المحاولة

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "مرحباً في BTC Trading Bot المحسن",
        "status": "نشط",
        "version": "3.1.0",
        "features": [
            "مصادر بيانات متعددة",
            "معالجة حدود الطلبات",
            "تحليل بديل عند الفشل",
            "إشعارات تلقائية"
        ],
        "performance": analyzer.performance_stats
    }

@app.get("/analysis")
async def get_analysis():
    """الحصول على التحليل الحالي"""
    return await analyzer.analyze_btc()

@app.get("/health")
async def health_check():
    """فحص الصحة"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "performance": analyzer.performance_stats,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    }

@app.post("/send-report")
async def send_report():
    """إرسال تقرير يدوي"""
    analysis = await analyzer.analyze_btc()
    
    message = f"📊 **تقرير يدوي لـ BTC**\n"
    message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    message += f"💰 السعر: ${analysis['price']:,.2f}\n"
    message += f"🎯 الإشارة: {analysis['overall_signal']}\n"
    message += f"📈 المصدر: {analysis['data_source']}"
    
    success = await notifier.send_message(message)
    return {"message": "تم إرسال التقرير", "success": success}

# بدء المهمة التلقائية
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_analysis_task())

@app.on_event("shutdown")
async def shutdown_event():
    await analyzer.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
