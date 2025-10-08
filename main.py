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
from scipy.signal import find_peaks

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

app = FastAPI(title="Crypto Market Phase Bot", version="10.0.0")

# إعدادات واقعية
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))
CONFIDENCE_THRESHOLD = 0.70  # 70% عتبة واقعية

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

class RealisticMarketAnalyzer:
    """محلل سوق واقعي يعتمد على تحليل تقليدي موثوق"""
    
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل واقعي للسوق يعتمد على منهجية متحفظة"""
        if len(prices) < 30:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}}
        
        try:
            df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
            
            # مؤشرات أساسية فقط
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # RSI بسيط
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # الحجم النسبي
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            latest = df.iloc[-1]
            prev_3 = df.iloc[-3] if len(df) > 3 else df.iloc[0]
            prev_10 = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            return RealisticMarketAnalyzer._conservative_analysis(latest, prev_3, prev_10, df)
            
        except Exception as e:
            safe_log_error(f"خطأ في التحليل: {e}", "N/A", "analyzer")
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار", "indicators": {}}
    
    @staticmethod
    def _conservative_analysis(latest, prev_3, prev_10, df) -> Dict[str, Any]:
        """تحليل متحفظ واقعي"""
        
        current_price = latest['close']
        price_change_3d = (current_price - prev_3['close']) / prev_3['close']
        price_change_10d = (current_price - prev_10['close']) / prev_10['close']
        
        # 🔍 تحليل الاتجاه الأساسي (الأهم)
        trend_strength = RealisticMarketAnalyzer._calculate_trend_strength(df)
        
        # 🔍 تحليل الزخم
        momentum_strength = RealisticMarketAnalyzer._calculate_momentum(latest, prev_3, prev_10)
        
        # 🔍 تحليل القوة الشرائية/البيعية
        volume_strength = RealisticMarketAnalyzer._calculate_volume_strength(latest, df)
        
        # 🔍 تحليل المؤشرات الفنية
        indicator_strength = RealisticMarketAnalyzer._calculate_indicator_strength(latest)
        
        # 🎯 تحديد المرحلة بناءً على تحليل متكامل
        phase, confidence = RealisticMarketAnalyzer._determine_phase_conservative(
            trend_strength, momentum_strength, volume_strength, indicator_strength,
            price_change_3d, price_change_10d, latest
        )
        
        # 🎯 توصية واقعية
        action = RealisticMarketAnalyzer._get_realistic_action(phase, confidence, current_price)
        
        return {
            "phase": phase,
            "confidence": round(confidence, 2),
            "action": action,
            "indicators": {
                "rsi": round(latest['rsi'], 1) if not pd.isna(latest['rsi']) else 50,
                "volume_ratio": round(latest['volume_ratio'], 2) if not pd.isna(latest['volume_ratio']) else 1.0,
                "macd_hist": round(latest['macd_hist'], 4) if not pd.isna(latest['macd_hist']) else 0.0,
                "trend": "صاعد" if latest['sma_20'] > latest['sma_50'] else "هابط",
                "price_change_3d": f"{price_change_3d*100:+.1f}%",
                "price_change_10d": f"{price_change_10d*100:+.1f}%",
                "momentum": "قوي" if momentum_strength > 0.7 else "ضعيف" if momentum_strength < 0.3 else "متوسط"
            }
        }
    
    @staticmethod
    def _calculate_trend_strength(df) -> float:
        """حساب قوة الاتجاه بشكل واقعي"""
        try:
            # اتجاه المتوسطات
            sma_trend = 1.0 if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else 0.0
            
            # اتجاه الأسعار الأخيرة
            recent_prices = df['close'].iloc[-5:]
            price_trend = 1.0 if recent_prices.iloc[-1] > recent_prices.iloc[0] else 0.0
            
            # قوة الاتجاه بناءً على استقراره
            trend_stability = min(abs(df['close'].iloc[-10:].pct_change().std() * 100), 1.0)
            
            return (sma_trend * 0.4 + price_trend * 0.4 + (1 - trend_stability) * 0.2)
        except:
            return 0.5
    
    @staticmethod
    def _calculate_momentum(latest, prev_3, prev_10) -> float:
        """حساب الزخم بشكل واقعي"""
        try:
            # زخم قصير المدى (3 أيام)
            short_momentum = 1.0 if latest['close'] > prev_3['close'] else 0.0
            
            # زخم طويل المدى (10 أيام)
            long_momentum = 1.0 if latest['close'] > prev_10['close'] else 0.0
            
            # قوة RSI
            rsi_strength = 0.0
            if not pd.isna(latest['rsi']):
                if latest['rsi'] > 60:
                    rsi_strength = 0.8
                elif latest['rsi'] < 40:
                    rsi_strength = 0.2
                else:
                    rsi_strength = 0.5
            
            return (short_momentum * 0.3 + long_momentum * 0.3 + rsi_strength * 0.4)
        except:
            return 0.5
    
    @staticmethod
    def _calculate_volume_strength(latest, df) -> float:
        """حساب قوة الحجم"""
        try:
            volume_ratio = latest['volume_ratio']
            if pd.isna(volume_ratio):
                return 0.5
                
            if volume_ratio > 1.5:
                return 0.8  # حجم عالي
            elif volume_ratio > 1.2:
                return 0.6  # حجم فوق المتوسط
            elif volume_ratio > 0.8:
                return 0.5  # حجم طبيعي
            else:
                return 0.3  # حجم منخفض
        except:
            return 0.5
    
    @staticmethod
    def _calculate_indicator_strength(latest) -> float:
        """حساب قوة المؤشرات الفنية"""
        try:
            # قوة MACD
            macd_strength = 0.5
            if not pd.isna(latest['macd_hist']):
                if latest['macd_hist'] > 0.01:
                    macd_strength = 0.8
                elif latest['macd_hist'] < -0.01:
                    macd_strength = 0.2
                else:
                    macd_strength = 0.5
            
            # قوة RSI
            rsi_strength = 0.5
            if not pd.isna(latest['rsi']):
                if 40 <= latest['rsi'] <= 60:
                    rsi_strength = 0.7  # RSI في منطقة متوازنة
                elif 30 <= latest['rsi'] <= 70:
                    rsi_strength = 0.5  # RSI في منطقة مقبولة
                else:
                    rsi_strength = 0.3  # RSI في منطقة متطرفة
            
            return (macd_strength * 0.6 + rsi_strength * 0.4)
        except:
            return 0.5
    
    @staticmethod
    def _determine_phase_conservative(trend, momentum, volume, indicators, change_3d, change_10d, latest) -> Tuple[str, float]:
        """تحديد المرحلة بمنهجية متحفظة"""
        
        # 🎯 شروط صارمة لكل مرحلة
        
        # شراء قوي (شروط صارمة)
        if (trend > 0.7 and momentum > 0.7 and volume > 0.6 and 
            indicators > 0.6 and change_3d > 0.02 and change_10d > 0.05):
            confidence = min((trend + momentum + volume + indicators) / 4 * 0.9, 0.85)
            return "صعود", confidence
        
        # بيع قوي (شروط صارمة)
        elif (trend < 0.3 and momentum < 0.3 and volume > 0.6 and 
              indicators < 0.4 and change_3d < -0.02 and change_10d < -0.05):
            confidence = min(( (1-trend) + (1-momentum) + volume + (1-indicators) ) / 4 * 0.9, 0.85)
            return "هبوط", confidence
        
        # تجميع (تراكم)
        elif (trend > 0.5 and momentum < 0.6 and volume < 0.7 and 
              indicators > 0.4 and abs(change_3d) < 0.05):
            confidence = (trend + (1-momentum) + (1-volume) + indicators) / 4 * 0.8
            return "تجميع", min(confidence, 0.75)
        
        # توزيع
        elif (trend < 0.6 and momentum > 0.4 and volume > 0.7 and 
              indicators < 0.6 and change_10d > 0.08):
            confidence = ((1-trend) + momentum + volume + (1-indicators)) / 4 * 0.8
            return "توزيع", min(confidence, 0.75)
        
        # اتجاه صاعد ضعيف
        elif trend > 0.6 and momentum > 0.5:
            confidence = (trend + momentum) / 2 * 0.7
            return "صعود", min(confidence, 0.65)
        
        # اتجاه هابط ضعيف
        elif trend < 0.4 and momentum < 0.5:
            confidence = ((1-trend) + (1-momentum)) / 2 * 0.7
            return "هبوط", min(confidence, 0.65)
        
        else:
            # غير محدد - معظم الحالات
            max_component = max(trend, momentum, volume, indicators)
            confidence = max_component * 0.5
            return "غير محدد", min(confidence, 0.5)
    
    @staticmethod
    def _get_realistic_action(phase: str, confidence: float, current_price: float) -> str:
        """توصيات واقعية ومحافظة"""
        
        if confidence > 0.75:
            if phase == "صعود":
                return "🟢 شراء - ثقة عالية"
            elif phase == "هبوط":
                return "🔴 بيع - مخاطر عالية"
            else:
                return "⚪ انتظار - إشارة قوية"
        
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
    """إشعارات واقعية"""
    
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
        
        # 🔴 منع الإشعارات غير الواقعية
        if current_confidence > 0.85:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: ثقة غير واقعية {current_confidence*100}%", coin, "reality_check")
            return False
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: ثقة واقعية {current_confidence*100}%", coin, "filter")
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        message = f"📊 **{coin.upper()} - {phase}**\n\n"
        message += f"💰 **السعر:** ${price:,.2f}\n"
        message += f"🎯 **الثقة:** {confidence*100}%\n"
        message += f"⚡ **التوصية:** {action}\n\n"
        
        message += f"🔍 **التحليل:**\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• الحجم: {indicators['volume_ratio']}x\n"
        message += f"• MACD: {indicators['macd_hist']}\n"
        message += f"• الاتجاه: {indicators['trend']}\n"
        message += f"• تغير 3 أيام: {indicators['price_change_3d']}\n"
        message += f"• الزخم: {indicators['momentum']}\n\n"
        
        message += f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ v10.0 - مرشح: {self.confidence_threshold*100}%"

        chart_base64 = self._generate_simple_chart(prices, coin)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                safe_log_info(f"تم إرسال إشعار واقعي لـ {coin}", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _generate_simple_chart(self, prices: List[float], coin: str) -> str:
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(prices, color='blue', linewidth=1.5)
            plt.title(f"{coin.upper()} - السعر")
            plt.xlabel("الفترة")
            plt.ylabel("السعر (USD)")
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
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
                'parse_mode': 'HTML'
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

class CryptoDataFetcher:
    """جلب بيانات بسيط"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.analyzer = RealisticMarketAnalyzer()
        self.cache = {}

    async def get_coin_data(self, coin_data: Dict[str, str]) -> Dict[str, Any]:
        cache_key = f"{coin_data['coingecko_id']}_data"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < CACHE_TTL:
            return self.cache[cache_key]['data']
        
        try:
            data = await self._fetch_from_binance(coin_data['binance_symbol'])
            if not data.get('prices'):
                data = await self._fetch_from_coingecko(coin_data['coingecko_id'])
            
            if not data.get('prices'):
                raise ValueError("لا بيانات")
            
            phase_analysis = self.analyzer.analyze_market_phase(
                data['prices'], data['highs'], data['lows'], data['volumes']
            )
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'phase_analysis': phase_analysis,
                'prices': data['prices'],
                'timestamp': current_time,
                'source': data['source']
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            return result
                
        except Exception as e:
            safe_log_error(f"خطأ في جلب بيانات {coin_data['symbol']}: {e}", coin_data['symbol'], "data_fetcher")
            return {'price': 0, 'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}}, 'prices': [], 'timestamp': current_time, 'source': 'fallback'}

    async def _fetch_from_coingecko(self, coin_id: str) -> Dict[str, Any]:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=60&interval=daily"
        for attempt in range(2):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    prices = [item[1] for item in data.get('prices', [])][-60:]
                    volumes = [item[1] for item in data.get('total_volumes', [])][-60:]
                    highs = [p * 1.01 for p in prices]
                    lows = [p * 0.99 for p in prices]
                    return {'prices': prices, 'highs': highs, 'lows': lows, 'volumes': volumes, 'source': 'coingecko'}
                await asyncio.sleep(2 ** attempt)
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'coingecko_failed'}

    async def _fetch_from_binance(self, symbol: str) -> Dict[str, Any]:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=60"
        for attempt in range(2):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'prices': [float(item[4]) for item in data],
                        'highs': [float(item[2]) for item in data],
                        'lows': [float(item[3]) for item in data],
                        'volumes': [float(item[5]) for item in data],
                        'source': 'binance'
                    }
                await asyncio.sleep(2 ** attempt)
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    async def close(self):
        await self.client.aclose()

# التهيئة
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def market_monitoring_task():
    """مراقبة واقعية"""
    safe_log_info(f"بدء المراقبة - عتبة واقعية: {CONFIDENCE_THRESHOLD*100}%", "all", "monitoring")
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    data = await data_fetcher.get_coin_data(coin_data)
                    phase_analysis = data['phase_analysis']
                    
                    safe_log_info(f"{coin_key}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']*100}%)", coin_key, "monitoring")
                    
                    if phase_analysis['confidence'] >= CONFIDENCE_THRESHOLD:
                        success = await notifier.send_phase_alert(coin_key, phase_analysis, data['price'], data['prices'])
                        if success:
                            safe_log_info(f"✅ تم إرسال إشعار واقعي لـ {coin_key}", coin_key, "monitoring")
                    else:
                        safe_log_info(f"⏭️ تخطي {coin_key}: ثقة غير كافية", coin_key, "monitoring")
                    
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    safe_log_error(f"خطأ في {coin_key}: {e}", coin_key, "monitoring")
                    continue
                    
            await asyncio.sleep(600)  # 10 دقائق بين الدورات
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(120)

@app.get("/")
async def root():
    return {"message": "بوت تحليل واقعي", "version": "10.0.0", "confidence_threshold": CONFIDENCE_THRESHOLD}

@app.get("/phase/{coin}")
async def get_coin_phase(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    return {"coin": coin, "price": data['price'], "phase_analysis": data['phase_analysis']}

@app.get("/status")
async def status():
    return {
        "status": "نشط - تحليل واقعي", 
        "version": "10.0.0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "cache_size": len(data_fetcher.cache)
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info(f"بدء التشغيل - v10.0.0 - تحليل واقعي", "system", "startup")
    asyncio.create_task(market_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف البوت", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
