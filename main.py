from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import math
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from logging.handlers import RotatingFileHandler

# إعداد التسجيل
logger = logging.getLogger("crypto_bot")
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

try:
    file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"تعذر إنشاء ملف التسجيل: {e}")

logger.propagate = False
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

app = FastAPI(title="Crypto Market Phase Bot", version="10.1.0")

# إعدادات واقعية
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 دقائق لتحديث أكثر تواتراً
CONFIDENCE_THRESHOLD = 0.60  # 60% عتبة أكثر واقعية

SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL'},
    'ada': {'name': 'Cardano', 'coingecko_id': 'cardano', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA'},
    'xrp': {'name': 'XRP', 'coingecko_id': 'ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP'},
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT'}
}

def safe_log_info(message: str, coin: str = "system", source: str = "app"):
    try:
        logger.info(f"{message} - Coin: {coin} - Source: {source}")
    except Exception as e:
        print(f"خطأ في التسجيل: {e} - الرسالة: {message}")

def safe_log_error(message: str, coin: str = "system", source: str = "app"):
    try:
        logger.error(f"{message} - Coin: {coin} - Source: {source}")
    except Exception as e:
        print(f"خطأ في تسجيل الخطأ: {e} - الرسالة: {message}")

class AccurateMarketAnalyzer:
    """محلل سوق دقيق يعتمد على بيانات Binance مباشرة"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """حساب RSI بدقة"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean().dropna().values
        avg_losses = pd.Series(losses).rolling(period).mean().dropna().values
        
        if len(avg_gains) == 0 or len(avg_losses) == 0:
            return 50.0
        
        rs = avg_gains[-1] / (avg_losses[-1] + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return min(max(rsi, 0), 100)

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        """حساب MACD بدقة"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
        
        macd_line = ema_12[-1] - ema_26[-1]
        signal_line = pd.Series([ema_12[i] - ema_26[i] for i in range(len(prices))]).ewm(span=9, adjust=False).mean().values[-1]
        histogram = macd_line - signal_line
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4),
            'histogram': round(histogram, 4)
        }

    @staticmethod
    def calculate_moving_averages(prices: List[float]) -> Dict[str, float]:
        """حساب المتوسطات المتحركة"""
        if len(prices) < 50:
            current_price = prices[-1] if prices else 0
            return {'sma_20': current_price, 'sma_50': current_price, 'ema_9': current_price, 'ema_21': current_price}
        
        sma_20 = pd.Series(prices).rolling(20).mean().values[-1]
        sma_50 = pd.Series(prices).rolling(50).mean().values[-1]
        ema_9 = pd.Series(prices).ewm(span=9, adjust=False).mean().values[-1]
        ema_21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values[-1]
        
        return {
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'ema_9': round(ema_9, 2),
            'ema_21': round(ema_21, 2)
        }

    @staticmethod
    def analyze_market_phase(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل دقيق للسوق يعتمد على بيانات حديثة"""
        if len(prices) < 50:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}}
        
        try:
            current_price = prices[-1]
            
            # حساب المؤشرات الفنية
            rsi = AccurateMarketAnalyzer.calculate_rsi(prices)
            macd_data = AccurateMarketAnalyzer.calculate_macd(prices)
            ma_data = AccurateMarketAnalyzer.calculate_moving_averages(prices)
            
            # حساب التغيرات السعرية
            price_change_24h = ((current_price - prices[-24]) / prices[-24] * 100) if len(prices) >= 24 else 0
            price_change_3d = ((current_price - prices[-3]) / prices[-3] * 100) if len(prices) >= 3 else 0
            price_change_7d = ((current_price - prices[-7]) / prices[-7] * 100) if len(prices) >= 7 else 0
            
            # حساب قوة الحجم
            volume_ratio = (volumes[-1] / pd.Series(volumes).rolling(20).mean().values[-1]) if len(volumes) >= 20 else 1.0
            
            # تحليل الاتجاه
            trend_strength = AccurateMarketAnalyzer._calculate_trend_strength(prices, ma_data)
            
            # تحليل الزخم
            momentum_strength = AccurateMarketAnalyzer._calculate_momentum_strength(rsi, macd_data, price_change_24h)
            
            # تحديد المرحلة
            phase, confidence = AccurateMarketAnalyzer._determine_market_phase(
                trend_strength, momentum_strength, rsi, macd_data, ma_data, current_price
            )
            
            # التوصية
            action = AccurateMarketAnalyzer._get_trading_action(phase, confidence, rsi)
            
            return {
                "phase": phase,
                "confidence": round(confidence, 2),
                "action": action,
                "indicators": {
                    "rsi": round(rsi, 1),
                    "volume_ratio": round(volume_ratio, 2),
                    "macd_hist": macd_data['histogram'],
                    "macd_line": macd_data['macd'],
                    "macd_signal": macd_data['signal'],
                    "ema_9": ma_data['ema_9'],
                    "ema_21": ma_data['ema_21'],
                    "sma_20": ma_data['sma_20'],
                    "sma_50": ma_data['sma_50'],
                    "trend": "صاعد" if ma_data['ema_9'] > ma_data['ema_21'] else "هابط",
                    "price_change_24h": f"{price_change_24h:+.1f}%",
                    "price_change_3d": f"{price_change_3d:+.1f}%",
                    "price_change_7d": f"{price_change_7d:+.1f}%",
                    "momentum": "قوي" if momentum_strength > 0.7 else "ضعيف" if momentum_strength < 0.3 else "متوسط"
                }
            }
            
        except Exception as e:
            safe_log_error(f"خطأ في التحليل الدقيق: {e}", "N/A", "analyzer")
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار", "indicators": {}}

    @staticmethod
    def _calculate_trend_strength(prices: List[float], ma_data: Dict[str, float]) -> float:
        """حساب قوة الاتجاه"""
        try:
            # اتجاه المتوسطات
            ema_trend = 1.0 if ma_data['ema_9'] > ma_data['ema_21'] else 0.0
            sma_trend = 1.0 if ma_data['sma_20'] > ma_data['sma_50'] else 0.0
            
            # اتجاه الأسعار الأخيرة (آخر 10 فترات)
            recent_prices = prices[-10:]
            price_trend = 1.0 if recent_prices[-1] > recent_prices[0] else 0.0
            
            # استقرار الاتجاه
            trend_stability = min(abs(pd.Series(prices[-20:]).pct_change().std() * 100), 2.0) / 2.0
            
            return (ema_trend * 0.4 + sma_trend * 0.3 + price_trend * 0.2 + (1 - trend_stability) * 0.1)
        except:
            return 0.5

    @staticmethod
    def _calculate_momentum_strength(rsi: float, macd_data: Dict[str, float], price_change_24h: float) -> float:
        """حساب قوة الزخم"""
        try:
            # زخم RSI
            rsi_momentum = 0.0
            if rsi > 70:
                rsi_momentum = 0.9  # شراء مفرط
            elif rsi > 60:
                rsi_momentum = 0.7  # زخم صاعد قوي
            elif rsi > 50:
                rsi_momentum = 0.6  # زخم صاعد
            elif rsi > 40:
                rsi_momentum = 0.4  # زخم هابط
            elif rsi > 30:
                rsi_momentum = 0.3  # زخم هابط قوي
            else:
                rsi_momentum = 0.1  # بيع مفرط

            # زخم MACD
            macd_momentum = 0.5
            if macd_data['histogram'] > 0.01:
                macd_momentum = 0.8
            elif macd_data['histogram'] > 0:
                macd_momentum = 0.6
            elif macd_data['histogram'] > -0.01:
                macd_momentum = 0.4
            else:
                macd_momentum = 0.2

            # زخم السعر
            price_momentum = 0.5
            if price_change_24h > 3:
                price_momentum = 0.8
            elif price_change_24h > 1:
                price_momentum = 0.6
            elif price_change_24h > -1:
                price_momentum = 0.5
            elif price_change_24h > -3:
                price_momentum = 0.4
            else:
                price_momentum = 0.2

            return (rsi_momentum * 0.4 + macd_momentum * 0.4 + price_momentum * 0.2)
        except:
            return 0.5

    @staticmethod
    def _determine_market_phase(trend_strength: float, momentum_strength: float, rsi: float, 
                               macd_data: Dict[str, float], ma_data: Dict[str, float], 
                               current_price: float) -> Tuple[str, float]:
        """تحديد مرحلة السوق بدقة"""
        
        # صعود قوي
        if (trend_strength > 0.7 and momentum_strength > 0.7 and 
            rsi > 60 and macd_data['histogram'] > 0 and 
            current_price > ma_data['sma_20']):
            confidence = min((trend_strength + momentum_strength) / 2 * 0.9, 0.85)
            return "صعود", confidence
        
        # هبوط قوي
        elif (trend_strength < 0.3 and momentum_strength < 0.3 and 
              rsi < 40 and macd_data['histogram'] < 0 and 
              current_price < ma_data['sma_20']):
            confidence = min(((1 - trend_strength) + (1 - momentum_strength)) / 2 * 0.9, 0.85)
            return "هبوط", confidence
        
        # صعود معتدل
        elif (trend_strength > 0.6 and momentum_strength > 0.5 and 
              current_price > ma_data['ema_9']):
            confidence = (trend_strength + momentum_strength) / 2 * 0.7
            return "صعود", min(confidence, 0.75)
        
        # هبوط معتدل
        elif (trend_strength < 0.4 and momentum_strength < 0.5 and 
              current_price < ma_data['ema_9']):
            confidence = ((1 - trend_strength) + (1 - momentum_strength)) / 2 * 0.7
            return "هبوط", min(confidence, 0.75)
        
        # توطيد (تجميع/توزيع)
        elif (0.4 <= trend_strength <= 0.6 and 
              0.4 <= momentum_strength <= 0.6 and 
              40 <= rsi <= 60):
            confidence = 0.5
            if current_price > ma_data['sma_50']:
                return "تجميع", confidence
            else:
                return "توزيع", confidence
        
        else:
            # غير محدد
            confidence = max(trend_strength, momentum_strength) * 0.5
            return "غير محدد", min(confidence, 0.5)

    @staticmethod
    def _get_trading_action(phase: str, confidence: float, rsi: float) -> str:
        """توصيات تداول واقعية"""
        
        if confidence > 0.75:
            if phase == "صعود" and rsi < 70:
                return "🟢 شراء - ثقة عالية"
            elif phase == "هبوط" and rsi > 30:
                return "🔴 بيع - مخاطر عالية"
            else:
                return "⚪ انتظار - إشارة قوية ولكن RSI متطرف"
        
        elif confidence > 0.65:
            if phase == "صعود":
                return "🟢 مراقبة للشراء"
            elif phase == "هبوط":
                return "🔴 مراقبة للبيع"
            elif phase == "تجميع":
                return "🟡 استعداد للشراء"
            elif phase == "توزيع":
                return "🟠 استعداد للبيع"
            else:
                return "⚪ انتظار"
        
        elif confidence > 0.55:
            if phase == "صعود":
                return "🟢 إشارة شراء ضعيفة"
            elif phase == "هبوط":
                return "🔴 إشارة بيع ضعيفة"
            else:
                return "⚪ انتظار - إشارات ضعيفة"
        
        else:
            return "⚪ انتظار - عدم وضوح"

class TelegramNotifier:
    """إشعارات دقيقة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        safe_log_info(f"تهيئة بعتبة ثقة: {self.confidence_threshold*100}%", "system", "config")

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        current_confidence = analysis["confidence"]
        
        safe_log_info(f"فحص {coin}: الثقة {current_confidence*100}% vs العتبة {self.confidence_threshold*100}%", coin, "filter")
        
        if current_confidence < self.confidence_threshold:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: الثقة غير كافية", coin, "filter")
            return False
        
        # فحص واقعية المؤشرات
        rsi = analysis["indicators"]["rsi"]
        if rsi > 80 or rsi < 20:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: RSI متطرف {rsi}", coin, "reality_check")
            return False
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: ثقة واقعية {current_confidence*100}%", coin, "filter")
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        message = f"📊 **{coin.upper()} - {phase}**\n\n"
        message += f"💰 **السعر:** ${price:,.2f}\n"
        message += f"🎯 **الثقة:** {confidence*100:.1f}%\n"
        message += f"⚡ **التوصية:** {action}\n\n"
        
        message += f"🔍 **التحليل:**\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• الحجم: {indicators['volume_ratio']}x\n"
        message += f"• MACD: {indicators['macd_hist']:.3f}\n"
        message += f"• الاتجاه: {indicators['trend']}\n"
        message += f"• تغير 24س: {indicators['price_change_24h']}\n"
        message += f"• الزخم: {indicators['momentum']}\n"
        message += f"• EMA(9): {indicators['ema_9']:.2f}\n"
        message += f"• EMA(21): {indicators['ema_21']:.2f}\n\n"
        
        message += f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ v10.1 - مرشح: {self.confidence_threshold*100}%"

        chart_base64 = self._generate_accurate_chart(prices, coin, indicators)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                safe_log_info(f"تم إرسال إشعار دقيق لـ {coin}", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _generate_accurate_chart(self, prices: List[float], coin: str, indicators: Dict[str, Any]) -> str:
        try:
            plt.figure(figsize=(10, 6))
            
            # رسم السعر
            plt.plot(prices[-100:], color='blue', linewidth=2, label='السعر')
            
            # رسم المتوسطات إذا كانت متوفرة
            if len(prices) >= 21:
                ema_9 = pd.Series(prices).ewm(span=9, adjust=False).mean().values[-100:]
                ema_21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values[-100:]
                plt.plot(ema_9, color='orange', linewidth=1, label='EMA(9)')
                plt.plot(ema_21, color='red', linewidth=1, label='EMA(21)')
            
            plt.title(f"{coin.upper()} - التحليل الدقيق")
            plt.xlabel("الفترة")
            plt.ylabel("السعر (USDT)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            safe_log_error(f"خطأ في الرسم البياني: {e}", coin, "chart")
            return ""

    async def _send_photo_with_caption(self, caption: str, photo_base64: str) -> bool:
        if not self.token or not self.chat_id or not photo_base64:
            return False
            
        try:
            if len(caption) > 1024:
                caption = caption[:1018] + "..."
                
            payload = {
                'chat_id': self.chat_id,
                'caption': caption,
                'parse_mode': 'Markdown'
            }
            
            files = {
                'photo': ('chart.png', base64.b64decode(photo_base64), 'image/png')
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendPhoto", data=payload, files=files, timeout=15.0)
                
            return response.status_code == 200
        except Exception as e:
            safe_log_error(f"خطأ في إرسال الصورة: {e}", "system", "telegram")
            return False

class AccurateDataFetcher:
    """جلب بيانات دقيق من Binance مباشرة"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.analyzer = AccurateMarketAnalyzer()
        self.cache = {}

    async def get_coin_data(self, coin_data: Dict[str, str]) -> Dict[str, Any]:
        cache_key = f"{coin_data['binance_symbol']}_accurate"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < CACHE_TTL:
            return self.cache[cache_key]['data']
        
        try:
            # استخدام Binance كمصدر رئيسي فقط
            data = await self._fetch_from_binance_accurate(coin_data['binance_symbol'])
            
            if not data.get('prices'):
                safe_log_error(f"فشل جلب بيانات من Binance لـ {coin_data['symbol']}", coin_data['symbol'], "data_fetcher")
                return self._get_fallback_data(current_time)
            
            phase_analysis = self.analyzer.analyze_market_phase(
                data['prices'], data['volumes']
            )
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'phase_analysis': phase_analysis,
                'prices': data['prices'],
                'highs': data['highs'],
                'lows': data['lows'],
                'volumes': data['volumes'],
                'timestamp': current_time,
                'source': 'binance_accurate'
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            safe_log_info(f"تم جلب بيانات دقيقة لـ {coin_data['symbol']} من Binance", coin_data['symbol'], "data_fetcher")
            return result
                
        except Exception as e:
            safe_log_error(f"خطأ في جلب بيانات {coin_data['symbol']}: {e}", coin_data['symbol'], "data_fetcher")
            return self._get_fallback_data(current_time)

    async def _fetch_from_binance_accurate(self, symbol: str) -> Dict[str, Any]:
        """جلب بيانات دقيقة من Binance بفترات مناسبة"""
        urls = [
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=168",  # 7 أيام بساعات
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=288"   # 24 ساعة ب5 دقائق
        ]
        
        for url in urls:
            for attempt in range(3):
                try:
                    response = await self.client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            return {
                                'prices': [float(item[4]) for item in data],  # سعر الإغلاق
                                'highs': [float(item[2]) for item in data],   # أعلى سعر
                                'lows': [float(item[3]) for item in data],    # أدنى سعر
                                'volumes': [float(item[5]) for item in data], # حجم التداول
                                'source': 'binance_accurate'
                            }
                    await asyncio.sleep(1 ** attempt)
                except Exception as e:
                    safe_log_error(f"خطأ في جلب البيانات من {url}: {e}", symbol, "binance_fetch")
                    await asyncio.sleep(1 ** attempt)
        
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    def _get_fallback_data(self, timestamp: float) -> Dict[str, Any]:
        """بيانات احتياطية عند الفشل"""
        return {
            'price': 0,
            'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}},
            'prices': [],
            'highs': [],
            'lows': [],
            'volumes': [],
            'timestamp': timestamp,
            'source': 'fallback'
        }

    async def close(self):
        await self.client.aclose()

# التهيئة
data_fetcher = AccurateDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def accurate_market_monitoring_task():
    """مراقبة دقيقة للسوق"""
    safe_log_info(f"بدء المراقبة الدقيقة - عتبة واقعية: {CONFIDENCE_THRESHOLD*100}%", "all", "monitoring")
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    data = await data_fetcher.get_coin_data(coin_data)
                    phase_analysis = data['phase_analysis']
                    
                    safe_log_info(f"{coin_key}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']*100:.1f}%, RSI: {phase_analysis['indicators'].get('rsi', 0)})", coin_key, "monitoring")
                    
                    if phase_analysis['confidence'] >= CONFIDENCE_THRESHOLD:
                        success = await notifier.send_phase_alert(coin_key, phase_analysis, data['price'], data['prices'])
                        if success:
                            safe_log_info(f"✅ تم إرسال إشعار دقيق لـ {coin_key}", coin_key, "monitoring")
                            # انتظار بعد إرسال إشعار ناجح لتجنب التكرار
                            await asyncio.sleep(10)
                    else:
                        safe_log_info(f"⏭️ تخطي {coin_key}: ثقة غير كافية", coin_key, "monitoring")
                    
                    await asyncio.sleep(2)  # انتظار قصير بين العملات
                    
                except Exception as e:
                    safe_log_error(f"خطأ في {coin_key}: {e}", coin_key, "monitoring")
                    continue
                    
            safe_log_info("اكتملت دورة المراقبة الدقيقة", "all", "monitoring")
            await asyncio.sleep(300)  # 5 دقائق بين الدورات
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(60)

@app.get("/")
async def root():
    return {
        "message": "بوت تحليل دقيق v10.1", 
        "version": "10.1.0", 
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "data_source": "Binance مباشرة"
    }

@app.get("/phase/{coin}")
async def get_coin_phase(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "العملة غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    return {
        "coin": coin, 
        "price": data['price'], 
        "phase_analysis": data['phase_analysis'],
        "data_source": data['source']
    }

@app.get("/status")
async def status():
    return {
        "status": "نشط - تحليل دقيق", 
        "version": "10.1.0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "data_source": "Binance مباشرة",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "cache_size": len(data_fetcher.cache),
        "cache_ttl": CACHE_TTL
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info(f"بدء التشغيل - v10.1.0 - تحليل دقيق من Binance", "system", "startup")
    asyncio.create_task(accurate_market_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف البوت الدقيق", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
