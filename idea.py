from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import math
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple, Optional
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from logging.handlers import RotatingFileHandler
import pytz

# =============================================================================
# الإعدادات الرئيسية - يمكن تعديلها بسهولة
# =============================================================================

# إعدادات التطبيق
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
PORT = int(os.getenv("PORT", 8000))

# إعدادات التداول
SCAN_INTERVAL = 900  # 15 دقيقة بين كل فحص (بالثواني)
CONFIDENCE_THRESHOLD = 50  # الحد الأدنى للنقاط لإرسال الإشعار

# الأصول والأطر الزمنية
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'bnb': {'name': 'Binance Coin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
}

TIMEFRAMES = ['1h', '15m']

# توقيت سوريا (GMT+3)
SYRIA_TZ = pytz.timezone('Asia/Damascus')

# أوقات الجلسات مع التوقيت السوري
TRADING_SESSIONS = {
    "asian": {"start": 0, "end": 8, "weight": 0.7},    # 03:00-11:00 توقيت سوريا
    "european": {"start": 8, "end": 16, "weight": 1.0}, # 11:00-19:00 توقيت سوريا
    "american": {"start": 16, "end": 24, "weight": 0.8} # 19:00-03:00 توقيت سوريا
}

# أوزان المؤشرات (من 100 نقطة)
INDICATOR_WEIGHTS = {
    "MOMENTUM": 40,      # RSI + Stochastic + MACD
    "PRICE_PATTERNS": 30, # أنماط الشموع + المتوسطات
    "LEVELS": 20,        # الدعم/المقاومة + فيبوناتشي
    "VOLUME": 10         # تحليل الحجم
}

# مستويات التنبيه
ALERT_LEVELS = {
    "LOW": {"min": 0, "max": 50, "emoji": "⚪", "send_alert": False},
    "MEDIUM": {"min": 51, "max": 70, "emoji": "🟡", "send_alert": True},
    "HIGH": {"min": 71, "max": 85, "emoji": "🟠", "send_alert": True},
    "STRONG": {"min": 86, "max": 100, "emoji": "🔴", "send_alert": True}
}

# =============================================================================
# نهاية الإعدادات الرئيسية
# =============================================================================

# إعداد التسجيل
logger = logging.getLogger("crypto_scanner")
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

try:
    file_handler = RotatingFileHandler("scanner.log", maxBytes=5*1024*1024, backupCount=3)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"تعذر إنشاء ملف التسجيل: {e}")

logger.propagate = False
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

app = FastAPI(title="Crypto Top/Bottom Scanner", version="1.0.0")

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

def get_syria_time():
    """الحصول على التوقيت السوري الحالي"""
    return datetime.now(SYRIA_TZ)

def get_session_weight():
    """الحصول على وزن الجلسة الحالية حسب التوقيت السوري"""
    current_time = get_syria_time()
    current_hour = current_time.hour
    
    for session, config in TRADING_SESSIONS.items():
        if config["start"] <= current_hour < config["end"]:
            return config["weight"]
    
    return 0.7  # افتراضي

def get_alert_level(score: int) -> Dict[str, Any]:
    """تحديد مستوى التنبيه بناء على النقاط"""
    for level, config in ALERT_LEVELS.items():
        if config["min"] <= score <= config["max"]:
            return {
                "level": level,
                "emoji": config["emoji"],
                "send_alert": config["send_alert"],
                "min": config["min"],
                "max": config["max"]
            }
    return ALERT_LEVELS["LOW"]

class AdvancedMarketAnalyzer:
    """محلل متقدم للقمم والقيعان"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """حساب RSI"""
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
    def calculate_stochastic(prices: List[float], period: int = 14) -> Dict[str, float]:
        """حساب Stochastic"""
        if len(prices) < period:
            return {'k': 50, 'd': 50}
        
        low_min = min(prices[-period:])
        high_max = max(prices[-period:])
        
        if high_max == low_min:
            k = 50
        else:
            k = 100 * ((prices[-1] - low_min) / (high_max - low_min))
        
        k_values = []
        for i in range(len(prices) - period + 1):
            period_low = min(prices[i:i+period])
            period_high = max(prices[i:i+period])
            if period_high != period_low:
                k_val = 100 * ((prices[i+period-1] - period_low) / (period_high - period_low))
                k_values.append(k_val)
            else:
                k_values.append(50)
        
        if len(k_values) >= 3:
            d = np.mean(k_values[-3:])
        else:
            d = k
        
        return {'k': round(k, 2), 'd': round(d, 2)}

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        """حساب MACD"""
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
    def detect_candle_pattern(prices: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """الكشف عن أنماط الشموع الانعكاسية"""
        if len(prices) < 3:
            return {"pattern": "none", "strength": 0, "description": "لا توجد بيانات كافية"}
        
        current_close = prices[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        prev_close = prices[-2]
        prev_high = highs[-2]
        prev_low = lows[-2]
        
        # حساب جسم الشمعة وذيلها
        current_body = abs(current_close - prices[-3])  # السعر الافتتاح (الشمعة السابقة close)
        current_upper_wick = current_high - max(current_close, prices[-3])
        current_lower_wick = min(current_close, prices[-3]) - current_low
        
        # نمط المطرقة (Hammer) - إشارة قاع
        is_hammer = (current_lower_wick > 2 * current_body and 
                    current_upper_wick < current_body * 0.5)
        
        # نمط النجم الساقط (Shooting Star) - إشارة قمة
        is_shooting_star = (current_upper_wick > 2 * current_body and 
                           current_lower_wick < current_body * 0.5)
        
        # نمط الابتلاع (Engulfing)
        is_bullish_engulfing = (current_close > prev_high and prices[-3] < prev_low)
        is_bearish_engulfing = (current_close < prev_low and prices[-3] > prev_high)
        
        if is_hammer:
            return {"pattern": "hammer", "strength": 8, "description": "مطرقة - إشارة قاع", "direction": "bottom"}
        elif is_shooting_star:
            return {"pattern": "shooting_star", "strength": 8, "description": "نجم ساقط - إشارة قمة", "direction": "top"}
        elif is_bullish_engulfing:
            return {"pattern": "bullish_engulfing", "strength": 7, "description": "ابتلاع صاعد - إشارة قاع", "direction": "bottom"}
        elif is_bearish_engulfing:
            return {"pattern": "bearish_engulfing", "strength": 7, "description": "ابتلاع هابط - إشارة قمة", "direction": "top"}
        else:
            return {"pattern": "none", "strength": 0, "description": "لا يوجد نمط واضح", "direction": "none"}

    @staticmethod
    def analyze_support_resistance(prices: List[float]) -> Dict[str, Any]:
        """تحليل مستويات الدعم والمقاومة"""
        if len(prices) < 20:
            return {"support": 0, "resistance": 0, "strength": 0}
        
        # استخدام أدنى وأعلى 20 شمعة
        recent_lows = min(prices[-20:])
        recent_highs = max(prices[-20:])
        current_price = prices[-1]
        
        # حساب القرب من المستويات
        distance_to_support = abs(current_price - recent_lows) / current_price
        distance_to_resistance = abs(current_price - recent_highs) / current_price
        
        strength = 0
        if distance_to_support < 0.02:  # within 2%
            strength = 8
            direction = "bottom"
        elif distance_to_resistance < 0.02:
            strength = 8
            direction = "top"
        else:
            strength = 0
            direction = "none"
            
        return {
            "support": recent_lows,
            "resistance": recent_highs,
            "strength": strength,
            "direction": direction,
            "current_price": current_price
        }

    @staticmethod
    def analyze_volume_trend(volumes: List[float]) -> Dict[str, Any]:
        """تحليل اتجاه الحجم"""
        if len(volumes) < 10:
            return {"trend": "stable", "strength": 0}
        
        recent_volume = np.mean(volumes[-5:])
        previous_volume = np.mean(volumes[-10:-5])
        
        volume_ratio = recent_volume / previous_volume
        
        if volume_ratio > 1.5:
            return {"trend": "strong_rising", "strength": 8}
        elif volume_ratio > 1.2:
            return {"trend": "rising", "strength": 6}
        elif volume_ratio < 0.7:
            return {"trend": "strong_falling", "strength": 8}
        elif volume_ratio < 0.9:
            return {"trend": "falling", "strength": 6}
        else:
            return {"trend": "stable", "strength": 3}

    @staticmethod
    def calculate_fibonacci_levels(prices: List[float]) -> Dict[str, float]:
        """حساب مستويات فيبوناتشي"""
        if len(prices) < 20:
            return {}
        
        high = max(prices[-20:])
        low = min(prices[-20:])
        current = prices[-1]
        
        diff = high - low
        
        levels = {
            '0.0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1.0': high
        }
        
        # إيجاد أقرب مستوى
        closest_level = None
        min_distance = float('inf')
        
        for level_name, level_price in levels.items():
            distance = abs(current - level_price) / current
            if distance < min_distance and distance < 0.02:  # within 2%
                min_distance = distance
                closest_level = level_name
        
        return {
            'levels': levels,
            'closest_level': closest_level,
            'distance': min_distance if closest_level else None
        }

    def analyze_market_condition(self, prices: List[float], volumes: List[float], 
                               highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """تحليل شامل لحالة السوق للكشف عن القمم والقيعان"""
        
        if len(prices) < 20:
            return self._get_empty_analysis()
        
        try:
            # حساب المؤشرات
            rsi = self.calculate_rsi(prices)
            stoch = self.calculate_stochastic(prices)
            macd = self.calculate_macd(prices)
            candle_pattern = self.detect_candle_pattern(prices, highs, lows)
            support_resistance = self.analyze_support_resistance(prices)
            volume_analysis = self.analyze_volume_trend(volumes)
            fib_levels = self.calculate_fibonacci_levels(prices)
            
            # تحليل القمة (Top)
            top_score = self._calculate_top_score(rsi, stoch, macd, candle_pattern, 
                                                support_resistance, volume_analysis, fib_levels)
            
            # تحليل القاع (Bottom)
            bottom_score = self._calculate_bottom_score(rsi, stoch, macd, candle_pattern,
                                                      support_resistance, volume_analysis, fib_levels)
            
            # تطبيق وزن الجلسة
            session_weight = get_session_weight()
            top_score = int(top_score * session_weight)
            bottom_score = int(bottom_score * session_weight)
            
            # تحديد الإشارة الأقوى
            strongest_signal = "top" if top_score > bottom_score else "bottom"
            strongest_score = max(top_score, bottom_score)
            
            return {
                "top_score": top_score,
                "bottom_score": bottom_score,
                "strongest_signal": strongest_signal,
                "strongest_score": strongest_score,
                "alert_level": get_alert_level(strongest_score),
                "indicators": {
                    "rsi": round(rsi, 2),
                    "stoch_k": stoch['k'],
                    "stoch_d": stoch['d'],
                    "macd_histogram": macd['histogram'],
                    "candle_pattern": candle_pattern,
                    "support_resistance": support_resistance,
                    "volume_trend": volume_analysis,
                    "fibonacci": fib_levels,
                    "session_weight": session_weight
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            safe_log_error(f"خطأ في تحليل السوق: {e}", "analyzer", "market_analysis")
            return self._get_empty_analysis()

    def _calculate_top_score(self, rsi: float, stoch: Dict, macd: Dict, 
                           candle_pattern: Dict, support_resistance: Dict,
                           volume_analysis: Dict, fib_levels: Dict) -> int:
        """حساب نقاط القمة"""
        score = 0
        
        # الزخم (40 نقطة)
        if rsi > 70: score += 15
        elif rsi > 60: score += 8
        
        if stoch['k'] > 80 and stoch['d'] > 80: score += 15
        elif stoch['k'] > 70 and stoch['d'] > 70: score += 8
        
        if macd['histogram'] < -0.01: score += 10
        elif macd['histogram'] < 0: score += 5
        
        # أنماط السعر (30 نقطة)
        if candle_pattern["direction"] == "top":
            score += candle_pattern["strength"]
        
        # اختراق المتوسطات (نقاط إضافية)
        score += 5  # قاعدة
        
        # المستويات (20 نقطة)
        if support_resistance["direction"] == "top":
            score += support_resistance["strength"]
        
        if fib_levels.get('closest_level') in ['0.618', '0.786', '1.0']:
            score += 8
        
        # الحجم (10 نقطة)
        if volume_analysis["trend"] in ["strong_rising", "rising"]:
            score += volume_analysis["strength"]
        
        return min(score, 100)

    def _calculate_bottom_score(self, rsi: float, stoch: Dict, macd: Dict,
                              candle_pattern: Dict, support_resistance: Dict,
                              volume_analysis: Dict, fib_levels: Dict) -> int:
        """حساب نقاط القاع"""
        score = 0
        
        # الزخم (40 نقطة)
        if rsi < 30: score += 15
        elif rsi < 40: score += 8
        
        if stoch['k'] < 20 and stoch['d'] < 20: score += 15
        elif stoch['k'] < 30 and stoch['d'] < 30: score += 8
        
        if macd['histogram'] > 0.01: score += 10
        elif macd['histogram'] > 0: score += 5
        
        # أنماط السعر (30 نقطة)
        if candle_pattern["direction"] == "bottom":
            score += candle_pattern["strength"]
        
        # اختراق المتوسطات (نقاط إضافية)
        score += 5  # قاعدة
        
        # المستويات (20 نقطة)
        if support_resistance["direction"] == "bottom":
            score += support_resistance["strength"]
        
        if fib_levels.get('closest_level') in ['0.0', '0.236', '0.382']:
            score += 8
        
        # الحجم (10 نقطة)
        if volume_analysis["trend"] in ["strong_rising", "rising"]:
            score += volume_analysis["strength"]
        
        return min(score, 100)

    def _get_empty_analysis(self) -> Dict[str, Any]:
        """تحليل افتراضي عند عدم وجود بيانات كافية"""
        return {
            "top_score": 0,
            "bottom_score": 0,
            "strongest_signal": "none",
            "strongest_score": 0,
            "alert_level": get_alert_level(0),
            "indicators": {},
            "timestamp": time.time()
        }

class TelegramNotifier:
    """إشعارات التليجرام للإشارات القوية"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.analyzer = AdvancedMarketAnalyzer()

    async def send_alert(self, coin: str, timeframe: str, analysis: Dict[str, Any], 
                        price: float, prices: List[float]) -> bool:
        """إرسال تنبيه لإشارة قوية"""
        
        alert_level = analysis["alert_level"]
        strongest_signal = analysis["strongest_signal"]
        strongest_score = analysis["strongest_score"]
        
        if not alert_level["send_alert"] or strongest_score < CONFIDENCE_THRESHOLD:
            return False
        
        try:
            # بناء الرسالة
            message = self._build_alert_message(coin, timeframe, analysis, price)
            
            # إرسال الرسالة
            async with httpx.AsyncClient() as client:
                payload = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                
                response = await client.post(f"{self.base_url}/sendMessage", 
                                           json=payload, timeout=10.0)
                
                if response.status_code == 200:
                    safe_log_info(f"تم إرسال إشعار لـ {coin} ({timeframe}) - {strongest_signal} - {strongest_score} نقطة", 
                                coin, "telegram")
                    return True
                else:
                    safe_log_error(f"فشل إرسال إشعار: {response.status_code}", coin, "telegram")
                    return False
                    
        except Exception as e:
            safe_log_error(f"خطأ في إرسال الإشعار: {e}", coin, "telegram")
            return False

    def _build_alert_message(self, coin: str, timeframe: str, analysis: Dict[str, Any], price: float) -> str:
        """بناء رسالة التنبيه"""
        
        alert_level = analysis["alert_level"]
        strongest_signal = analysis["strongest_signal"]
        strongest_score = analysis["strongest_score"]
        indicators = analysis["indicators"]
        
        # الرأس
        signal_emoji = "🔴" if strongest_signal == "top" else "🟢"
        message = f"{signal_emoji} **تنبيه {coin.upper()} - إطار {timeframe}** {signal_emoji}\n\n"
        
        # المعلومات الأساسية
        message += f"📊 **نوع الإشارة:** {'قمة 🔴' if strongest_signal == 'top' else 'قاع 🟢'}\n"
        message += f"🎯 **قوة الإشارة:** {alert_level['emoji']} **{strongest_score}/100**\n"
        message += f"💰 **السعر الحالي:** ${price:,.2f}\n"
        message += f"⏰ **التوقيت السوري:** {get_syria_time().strftime('%H:%M %d/%m/%Y')}\n\n"
        
        # المؤشرات
        message += "📈 **المؤشرات:**\n"
        
        if 'rsi' in indicators:
            rsi_emoji = "🔴" if indicators['rsi'] > 70 else "🟢" if indicators['rsi'] < 30 else "🟡"
            message += f"• {rsi_emoji} RSI: **{indicators['rsi']}**\n"
        
        if 'stoch_k' in indicators:
            stoch_emoji = "🔴" if indicators['stoch_k'] > 80 else "🟢" if indicators['stoch_k'] < 20 else "🟡"
            message += f"• {stoch_emoji} Stochastic: **K={indicators['stoch_k']}, D={indicators['stoch_d']}**\n"
        
        if 'macd_histogram' in indicators:
            macd_emoji = "🟢" if indicators['macd_histogram'] > 0 else "🔴"
            message += f"• {macd_emoji} MACD Hist: **{indicators['macd_histogram']:.4f}**\n"
        
        if 'candle_pattern' in indicators and indicators['candle_pattern']['pattern'] != 'none':
            message += f"• 🕯️ نمط الشموع: **{indicators['candle_pattern']['description']}**\n"
        
        if 'volume_trend' in indicators:
            volume_emoji = "📈" if "rising" in indicators['volume_trend']['trend'] else "📉"
            message += f"• {volume_emoji} الحجم: **{indicators['volume_trend']['trend']}**\n"
        
        if 'session_weight' in indicators:
            message += f"• ⚖️ وزن الجلسة: **{indicators['session_weight']*100}%**\n"
        
        message += f"\n⚡ **البوت:** ماسح القمم والقيعان v1.0"
        
        return message

class BinanceDataFetcher:
    """جلب البيانات من Binance"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.analyzer = AdvancedMarketAnalyzer()
        self.cache = {}

    async def get_coin_data(self, coin_data: Dict[str, str], timeframe: str) -> Dict[str, Any]:
        """جلب بيانات العملة للإطار الزمني المحدد"""
        
        cache_key = f"{coin_data['binance_symbol']}_{timeframe}"
        current_time = time.time()
        
        # التحقق من الكاش
        if cache_key in self.cache:
            cache_data = self.cache[cache_key]
            if current_time - cache_data['timestamp'] < 300:  # 5 دقائق كاش
                return cache_data['data']
        
        try:
            # جلب البيانات من Binance
            data = await self._fetch_binance_data(coin_data['binance_symbol'], timeframe)
            
            if not data.get('prices'):
                safe_log_error(f"فشل جلب بيانات {timeframe} لـ {coin_data['symbol']}", 
                             coin_data['symbol'], "data_fetcher")
                return self._get_fallback_data()
            
            # تحليل البيانات
            analysis = self.analyzer.analyze_market_condition(
                data['prices'], data['volumes'], data['highs'], data['lows']
            )
            
            result = {
                'price': data['prices'][-1],
                'analysis': analysis,
                'prices': data['prices'],
                'highs': data['highs'],
                'lows': data['lows'],
                'volumes': data['volumes'],
                'timestamp': current_time,
                'timeframe': timeframe
            }
            
            # تخزين في الكاش
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            
            safe_log_info(f"تم تحليل {coin_data['symbol']} ({timeframe}) - قمة: {analysis['top_score']} - قاع: {analysis['bottom_score']}", 
                         coin_data['symbol'], "analyzer")
            
            return result
                
        except Exception as e:
            safe_log_error(f"خطأ في جلب بيانات {coin_data['symbol']}: {e}", 
                         coin_data['symbol'], "data_fetcher")
            return self._get_fallback_data()

    async def _fetch_binance_data(self, symbol: str, interval: str) -> Dict[str, List[float]]:
        """جلب البيانات من Binance API"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=100"
        
        try:
            response = await self.client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'prices': [float(item[4]) for item in data],  # Close prices
                    'highs': [float(item[2]) for item in data],   # High prices
                    'lows': [float(item[3]) for item in data],    # Low prices
                    'volumes': [float(item[5]) for item in data]  # Volumes
                }
        except Exception as e:
            safe_log_error(f"خطأ في جلب البيانات من Binance: {e}", symbol, "binance")
        
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': []}

    def _get_fallback_data(self) -> Dict[str, Any]:
        """بيانات افتراضية عند فشل الجلب"""
        return {
            'price': 0,
            'analysis': self.analyzer._get_empty_analysis(),
            'prices': [],
            'highs': [],
            'lows': [],
            'volumes': [],
            'timestamp': time.time(),
            'timeframe': 'unknown'
        }

    async def close(self):
        await self.client.aclose()

# التهيئة العالمية
data_fetcher = BinanceDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def market_scanner_task():
    """المهمة الرئيسية للمسح الضوئي"""
    safe_log_info("بدء مهمة مسح السوق كل 15 دقيقة", "system", "scanner")
    
    while True:
        try:
            syria_time = get_syria_time()
            safe_log_info(f"بدء دورة المسح - التوقيت السوري: {syria_time.strftime('%H:%M %d/%m/%Y')}", 
                         "system", "scanner")
            
            alerts_sent = 0
            
            # مسح جميع العملات والأطر الزمنية
            for coin_key, coin_data in SUPPORTED_COINS.items():
                for timeframe in TIMEFRAMES:
                    try:
                        # جلب البيانات والتحليل
                        data = await data_fetcher.get_coin_data(coin_data, timeframe)
                        analysis = data['analysis']
                        
                        # إرسال التنبيه إذا كانت الإشارة قوية
                        if analysis["alert_level"]["send_alert"]:
                            success = await notifier.send_alert(
                                coin_key, timeframe, analysis, data['price'], data['prices']
                            )
                            if success:
                                alerts_sent += 1
                                await asyncio.sleep(2)  # فواصل بين الإشعارات
                        
                        await asyncio.sleep(1)  # فواصل بين الطلبات
                        
                    except Exception as e:
                        safe_log_error(f"خطأ في معالجة {coin_key} ({timeframe}): {e}", 
                                     coin_key, "scanner")
                        continue
            
            safe_log_info(f"اكتملت دورة المسح - تم إرسال {alerts_sent} تنبيه", 
                         "system", "scanner")
            
            # انتظار حتى الفحص التالي
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "system", "scanner")
            await asyncio.sleep(60)  # انتظار قصير عند الخطأ

async def health_check_task():
    """مهمة الفحص الصحي"""
    while True:
        try:
            # فحص بسيط للذاكرة والأداء
            current_time = time.time()
            cache_size = len(data_fetcher.cache)
            
            safe_log_info(f"الفحص الصحي - الكاش: {cache_size} - الوقت: {current_time}", 
                         "system", "health")
            
            await asyncio.sleep(300)  # فحص كل 5 دقائق
            
        except Exception as e:
            safe_log_error(f"خطأ في الفحص الصحي: {e}", "system", "health")
            await asyncio.sleep(60)

# endpoints للـ API
@app.get("/")
async def root():
    return {
        "message": "ماسح القمم والقيعان للكريبتو",
        "version": "1.0.0",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "timeframes": TIMEFRAMES,
        "scan_interval": f"{SCAN_INTERVAL} ثانية",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "syria_time": get_syria_time().strftime('%H:%M %d/%m/%Y')
    }

@app.get("/health")
async def health_check():
    return {
        "status": "نشط",
        "syria_time": get_syria_time().strftime('%H:%M %d/%m/%Y'),
        "session_weight": get_session_weight(),
        "cache_size": len(data_fetcher.cache)
    }

@app.get("/scan/{coin}")
async def scan_coin(coin: str, timeframe: str = "15m"):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(404, "الإطار الزمني غير مدعوم")
    
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data, timeframe)
    
    return {
        "coin": coin,
        "timeframe": timeframe,
        "price": data['price'],
        "analysis": data['analysis'],
        "syria_time": get_syria_time().strftime('%H:%M %d/%m/%Y')
    }

@app.get("/session-info")
async def get_session_info():
    current_time = get_syria_time()
    current_weight = get_session_weight()
    
    return {
        "syria_time": current_time.strftime('%H:%M %d/%m/%Y'),
        "current_hour": current_time.hour,
        "session_weight": current_weight,
        "trading_sessions": TRADING_SESSIONS
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info("بدء تشغيل ماسح القمم والقيعان", "system", "startup")
    safe_log_info(f"العملات المدعومة: {list(SUPPORTED_COINS.keys())}", "system", "config")
    safe_log_info(f"الأطر الزمنية: {TIMEFRAMES}", "system", "config")
    safe_log_info(f"فاصل المسح: {SCAN_INTERVAL} ثانية", "system", "config")
    safe_log_info(f"حد الثقة: {CONFIDENCE_THRESHOLD} نقطة", "system", "config")
    
    # بدء المهام
    asyncio.create_task(market_scanner_task())
    asyncio.create_task(health_check_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف ماسح السوق", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
