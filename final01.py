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

app = FastAPI(title="Crypto Trading Signals Bot", version="5.0.0")

# إعدادات البوت
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))
CONFIDENCE_THRESHOLD = 0.56

SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL'},
    'ada': {'name': 'Cardano', 'coingecko_id': 'cardano', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA'},
    'xrp': {'name': 'XRP', 'coingecko_id': 'ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP'},
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT'}
}

# تعريف مستويات الثقة مع تمييز واضح للإشارات
CONFIDENCE_LEVELS = {
    "NO_SIGNAL": {"min": 0.0, "max": 0.55, "emoji": "⚪", "color": "gray", "name": "لا إشارة", "send_alert": False},
    "MEDIUM": {"min": 0.56, "max": 0.65, "emoji": "🟡", "color": "gold", "name": "متوسط", "send_alert": True},
    "HIGH": {"min": 0.66, "max": 0.74, "emoji": "🟢", "color": "limegreen", "name": "قوي", "send_alert": True},
    "VERY_HIGH": {"min": 0.75, "max": 1.00, "emoji": "💚", "color": "darkgreen", "name": "قوي جداً", "send_alert": True}
}

# ألوان وإيموجيات مخصصة لكل نوع إشارة
SIGNAL_TYPES = {
    "BUY": {
        "emoji": "🟢", 
        "color": "green",
        "title": "شراء",
        "arrow": "📈",
        "action_emoji": "💹",
        "chart_color": "green",
        "signal_emoji": "✅"
    },
    "SELL": {
        "emoji": "🔴", 
        "color": "red",
        "title": "بيع",
        "arrow": "📉",
        "action_emoji": "🔄",
        "chart_color": "red",
        "signal_emoji": "⏹️"
    },
    "HOLD": {
        "emoji": "⚪", 
        "color": "gray",
        "title": "انتظار",
        "arrow": "➡️",
        "action_emoji": "⏳",
        "chart_color": "blue",
        "signal_emoji": "⚡"
    }
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

def get_confidence_level(confidence: float) -> Dict[str, Any]:
    """تحديد مستوى الثقة بناء على القيمة"""
    for level, config in CONFIDENCE_LEVELS.items():
        if config["min"] <= confidence <= config["max"]:
            return {
                "level": level,
                "name": config["name"],
                "emoji": config["emoji"],
                "color": config["color"],
                "min": config["min"],
                "max": config["max"],
                "send_alert": config["send_alert"]
            }
    return CONFIDENCE_LEVELS["NO_SIGNAL"]

def get_signal_type(signal: str) -> Dict[str, Any]:
    """الحصول على إعدادات نوع الإشارة"""
    return SIGNAL_TYPES.get(signal, SIGNAL_TYPES["HOLD"])

class TradingSignalAnalyzer:
    """محلل إشارات تداول دقيق مع تمييز واضح للإشارات"""
    
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
        """حساب المتوسطات المتحركة للإطار 5 دقائق"""
        if len(prices) < 50:
            current_price = prices[-1] if prices else 0
            return {
                'ema_8': current_price, 
                'ema_21': current_price, 
                'sma_20': current_price,
                'sma_50': current_price
            }
        
        ema_8 = pd.Series(prices).ewm(span=8, adjust=False).mean().values[-1]
        ema_21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values[-1]
        sma_20 = pd.Series(prices).rolling(20).mean().values[-1]
        sma_50 = pd.Series(prices).rolling(50).mean().values[-1]
        
        return {
            'ema_8': round(ema_8, 2),
            'ema_21': round(ema_21, 2),
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2)
        }

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
        """حساب Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return {'upper': current_price, 'middle': current_price, 'lower': current_price}
        
        sma = pd.Series(prices).rolling(period).mean().values[-1]
        std = pd.Series(prices).rolling(period).std().values[-1]
        
        return {
            'upper': round(sma + (std * 2), 2),
            'middle': round(sma, 2),
            'lower': round(sma - (std * 2), 2)
        }

    @staticmethod
    def calculate_stochastic(prices: List[float], period: int = 14) -> Dict[str, float]:
        """حساب Stochastic Oscillator"""
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
    def generate_trading_signals(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارات تداول دقيقة مع تمييز واضح"""
        if len(prices) < 50:
            return {
                "signal": "HOLD", 
                "confidence": 0, 
                "confidence_level": get_confidence_level(0),
                "action": "انتظار - بيانات غير كافية",
                "signal_type": get_signal_type("HOLD"),
                "indicators": {}
            }
        
        try:
            current_price = prices[-1]
            
            # حساب المؤشرات
            rsi = TradingSignalAnalyzer.calculate_rsi(prices)
            macd_data = TradingSignalAnalyzer.calculate_macd(prices)
            ma_data = TradingSignalAnalyzer.calculate_moving_averages(prices)
            bb_data = TradingSignalAnalyzer.calculate_bollinger_bands(prices)
            stoch_data = TradingSignalAnalyzer.calculate_stochastic(prices)
            
            # تحليل الحجم
            volume_trend = TradingSignalAnalyzer._analyze_volume_trend(volumes)
            
            # توليد الإشارات مع ثقة مفصلة
            signal, confidence = TradingSignalAnalyzer._generate_detailed_signals(
                current_price, rsi, macd_data, ma_data, bb_data, stoch_data, volume_trend
            )
            
            # الحصول على مستوى الثقة ونوع الإشارة
            confidence_level = get_confidence_level(confidence)
            signal_config = get_signal_type(signal)
            
            # التوصية بناء على الإشارة ومستوى الثقة
            action = TradingSignalAnalyzer._get_action_by_confidence_level(signal, confidence_level, rsi, signal_config)
            
            return {
                "signal": signal,
                "confidence": round(confidence, 2),
                "confidence_level": confidence_level,
                "action": action,
                "signal_type": signal_config,
                "indicators": {
                    "rsi": round(rsi, 1),
                    "macd_hist": macd_data['histogram'],
                    "macd_line": macd_data['macd'],
                    "macd_signal": macd_data['signal'],
                    "stoch_k": stoch_data['k'],
                    "stoch_d": stoch_data['d'],
                    "ema_8": ma_data['ema_8'],
                    "ema_21": ma_data['ema_21'],
                    "sma_20": ma_data['sma_20'],
                    "bb_upper": bb_data['upper'],
                    "bb_lower": bb_data['lower'],
                    "bb_position": "وسط" if bb_data['lower'] < current_price < bb_data['upper'] else "علوي" if current_price >= bb_data['upper'] else "سفلي",
                    "volume_trend": volume_trend,
                    "price_vs_ema8": "فوق" if current_price > ma_data['ema_8'] else "تحت",
                    "ema8_vs_ema21": "فوق" if ma_data['ema_8'] > ma_data['ema_21'] else "تحت",
                    "trend_strength": TradingSignalAnalyzer._calculate_trend_strength(prices),
                    "signal_strength": TradingSignalAnalyzer._calculate_signal_strength(signal, confidence)
                }
            }
            
        except Exception as e:
            safe_log_error(f"خطأ في توليد الإشارات: {e}", "N/A", "analyzer")
            return {
                "signal": "HOLD", 
                "confidence": 0, 
                "confidence_level": get_confidence_level(0),
                "action": "انتظار", 
                "signal_type": get_signal_type("HOLD"),
                "indicators": {}
            }

    @staticmethod
    def _analyze_volume_trend(volumes: List[float]) -> str:
        """تحليل اتجاه الحجم"""
        if len(volumes) < 10:
            return "مستقر"
        
        recent_volume = np.mean(volumes[-5:])
        previous_volume = np.mean(volumes[-10:-5])
        
        if recent_volume > previous_volume * 1.3:
            return "📈 متزايد بقوة"
        elif recent_volume > previous_volume * 1.1:
            return "📈 متزايد"
        elif recent_volume < previous_volume * 0.7:
            return "📉 متراجع بقوة"
        elif recent_volume < previous_volume * 0.9:
            return "📉 متراجع"
        else:
            return "➡️ مستقر"

    @staticmethod
    def _calculate_trend_strength(prices: List[float]) -> str:
        """حساب قوة الاتجاه"""
        if len(prices) < 20:
            return "ضعيف"
        
        price_change = ((prices[-1] - prices[-20]) / prices[-20]) * 100
        
        if abs(price_change) > 8:
            return "قوي جداً"
        elif abs(price_change) > 5:
            return "قوي"
        elif abs(price_change) > 2:
            return "متوسط"
        else:
            return "ضعيف"

    @staticmethod
    def _calculate_signal_strength(signal: str, confidence: float) -> str:
        """حساب قوة الإشارة"""
        if confidence > 0.75:
            return "قوية جداً"
        elif confidence > 0.65:
            return "قوية"
        elif confidence > 0.55:
            return "متوسطة"
        else:
            return "ضعيفة"

    @staticmethod
    def _generate_detailed_signals(current_price: float, rsi: float, macd_data: Dict[str, float], 
                                  ma_data: Dict[str, float], bb_data: Dict[str, float],
                                  stoch_data: Dict[str, float], volume_trend: str) -> Tuple[str, float]:
        """توليد إشارات تداول مفصلة مع ثقة دقيقة"""
        
        buy_points = 0
        sell_points = 0
        total_points = 0
        
        # 1. إشارة RSI (25 نقطة)
        if rsi < 30:
            buy_points += 25
        elif rsi < 40:
            buy_points += 20
        elif rsi < 50:
            buy_points += 10
        elif rsi > 70:
            sell_points += 25
        elif rsi > 60:
            sell_points += 20
        elif rsi > 50:
            sell_points += 10
        total_points += 25

        # 2. إشارة MACD (20 نقطة)
        if macd_data['histogram'] > 0.02 and macd_data['macd'] > macd_data['signal']:
            buy_points += 20
        elif macd_data['histogram'] > 0:
            buy_points += 15
        elif macd_data['histogram'] < -0.02 and macd_data['macd'] < macd_data['signal']:
            sell_points += 20
        elif macd_data['histogram'] < 0:
            sell_points += 15
        total_points += 20

        # 3. إشارة المتوسطات المتحركة (20 نقطة)
        if current_price > ma_data['ema_8'] > ma_data['ema_21']:
            buy_points += 20
        elif current_price > ma_data['ema_8']:
            buy_points += 12
        elif current_price < ma_data['ema_8'] < ma_data['ema_21']:
            sell_points += 20
        elif current_price < ma_data['ema_8']:
            sell_points += 12
        total_points += 20

        # 4. إشارة Bollinger Bands (15 نقطة)
        if current_price <= bb_data['lower']:
            buy_points += 15
        elif current_price >= bb_data['upper']:
            sell_points += 15
        elif bb_data['lower'] < current_price < bb_data['middle']:
            buy_points += 8
        elif bb_data['middle'] < current_price < bb_data['upper']:
            sell_points += 8
        total_points += 15

        # 5. إشارة Stochastic (10 نقطة)
        if stoch_data['k'] < 20 and stoch_data['d'] < 20:
            buy_points += 10
        elif stoch_data['k'] < 30 and stoch_data['d'] < 30:
            buy_points += 7
        elif stoch_data['k'] > 80 and stoch_data['d'] > 80:
            sell_points += 10
        elif stoch_data['k'] > 70 and stoch_data['d'] > 70:
            sell_points += 7
        total_points += 10

        # 6. إشارة الحجم (10 نقطة)
        if "متزايد" in volume_trend:
            if buy_points > sell_points:
                buy_points += 10
            elif sell_points > buy_points:
                sell_points += 10
        elif "متراجع" in volume_trend:
            if buy_points > sell_points:
                buy_points -= 5
            elif sell_points > buy_points:
                sell_points -= 5
        total_points += 10

        # حساب النسب النهائية
        buy_ratio = buy_points / total_points
        sell_ratio = sell_points / total_points
        
        # تحديد الإشارة النهائية مع ثقة محسنة
        min_threshold = 0.55
        
        if buy_ratio > min_threshold and buy_ratio > sell_ratio:
            confidence = buy_ratio
            return "BUY", confidence
        elif sell_ratio > min_threshold and sell_ratio > buy_ratio:
            confidence = sell_ratio
            return "SELL", confidence
        else:
            confidence = max(buy_ratio, sell_ratio)
            return "HOLD", confidence

    @staticmethod
    def _get_action_by_confidence_level(signal: str, confidence_level: Dict[str, Any], rsi: float, signal_config: Dict[str, Any]) -> str:
        """توصيات التداول بناء على مستوى الثقة مع تمييز واضح"""
        
        level_emoji = confidence_level["emoji"]
        signal_emoji = signal_config["emoji"]
        action_emoji = signal_config["action_emoji"]
        arrow_emoji = signal_config["arrow"]
        
        if signal == "BUY":
            if confidence_level["level"] == "VERY_HIGH":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} شراء قوي جداً - دخول فوري"
            elif confidence_level["level"] == "HIGH":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} شراء قوي - دخول جيد"
            elif confidence_level["level"] == "MEDIUM":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} شراء متوسط - انتظار تأكيد"
            else:
                return f"{signal_emoji} لا توجد إشارة شراء واضحة"
                
        elif signal == "SELL":
            if confidence_level["level"] == "VERY_HIGH":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} بيع قوي جداً - خروج فوري"
            elif confidence_level["level"] == "HIGH":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} بيع قوي - خروج جيد"
            elif confidence_level["level"] == "MEDIUM":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} بيع متوسط - انتظار تأكيد"
            else:
                return f"{signal_emoji} لا توجد إشارة بيع واضحة"
                
        else:  # HOLD
            if confidence_level["level"] in ["VERY_HIGH", "HIGH"]:
                return f"{signal_emoji} {action_emoji} {arrow_emoji} انتظار قوي - تجنب التداول"
            elif confidence_level["level"] == "MEDIUM":
                return f"{signal_emoji} {action_emoji} {arrow_emoji} انتظار - اتجاه غير واضح"
            else:
                return f"{signal_emoji} {action_emoji} لا توجد إشارة تداول واضحة"

class TelegramNotifier:
    """إشعارات التداول مع تمييز واضح للإشارات"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        safe_log_info(f"تهيئة بعتبة ثقة: {self.confidence_threshold*100}%", "system", "config")

    async def send_signal_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        current_confidence = analysis["confidence"]
        confidence_level = analysis["confidence_level"]
        signal_type = analysis["signal"]
        signal_config = analysis["signal_type"]
        
        safe_log_info(f"فحص {coin}: {signal_type} بثقة {current_confidence*100}% ({confidence_level['name']})", coin, "filter")
        
        if not confidence_level["send_alert"]:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: مستوى الثقة لا يسمح بالإرسال", coin, "filter")
            return False
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: {signal_type} بثقة {current_confidence*100}% ({confidence_level['name']})", coin, "filter")
        
        signal = analysis["signal"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        confidence_level = analysis["confidence_level"]
        signal_config = analysis["signal_type"]
        
        # بناء الرسالة مع تمييز واضح
        signal_emoji = signal_config["emoji"]
        level_emoji = confidence_level["emoji"]
        title = signal_config["title"]
        arrow = signal_config["arrow"]
        
        # رأس الرسالة مميز
        message = f"{signal_emoji} {arrow} **{coin.upper()} - إشارة {title}** {arrow}\n\n"
        
        # معلومات أساسية مع ألوان مميزة
        message += f"💰 **السعر الحالي:** ${price:,.2f}\n"
        message += f"🎯 **مستوى الثقة:** {level_emoji} {confidence_level['name']}\n"
        message += f"📊 **نسبة الثقة:** **{confidence*100:.1f}%**\n"
        message += f"⚡ **قوة الإشارة:** {indicators['signal_strength']}\n"
        message += f"🔍 **التوصية:** {action}\n\n"
        
        # المؤشرات الفنية مع تمييز
        message += f"📈 **المؤشرات الفنية:**\n"
        message += f"• {self._get_indicator_emoji('rsi', indicators['rsi'])} RSI: **{indicators['rsi']}**\n"
        message += f"• {self._get_indicator_emoji('macd', indicators['macd_hist'])} MACD: **{indicators['macd_hist']:.4f}**\n"
        message += f"• {self._get_indicator_emoji('stoch', indicators['stoch_k'])} Stochastic: K={indicators['stoch_k']}, D={indicators['stoch_d']}\n"
        message += f"• 📊 السعر/EMA8: **{indicators['price_vs_ema8']}**\n"
        message += f"• 📊 EMA8/EMA21: **{indicators['ema8_vs_ema21']}**\n"
        message += f"• 📊 Bollinger Band: **{indicators['bb_position']}**\n"
        message += f"• 💪 قوة الاتجاه: **{indicators['trend_strength']}**\n"
        message += f"• 🔊 اتجاه الحجم: **{indicators['volume_trend']}**\n\n"
        
        # المستويات الرئيسية
        message += f"🎯 **المستويات الرئيسية:**\n"
        message += f"• EMA8: `{indicators['ema_8']:.2f}`\n"
        message += f"• EMA21: `{indicators['ema_21']:.2f}`\n"
        message += f"• BB علوي: `{indicators['bb_upper']:.2f}`\n"
        message += f"• BB سفلي: `{indicators['bb_lower']:.2f}`\n\n"
        
        # التوقيع
        message += f"🕒 **الوقت:** {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ **الإطار:** 5 دقائق - v5.0"

        chart_base64 = self._generate_enhanced_chart(prices, coin, indicators, signal, confidence_level, signal_config)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                safe_log_info(f"تم إرسال إشعار إشارة لـ {coin} بمستوى {confidence_level['name']}", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _get_indicator_emoji(self, indicator: str, value: float) -> str:
        """الحصول على إيموجي المؤشر بناء على قيمته"""
        if indicator == 'rsi':
            if value < 30: return '🟢'
            elif value > 70: return '🔴'
            else: return '🟡'
        elif indicator == 'macd':
            if value > 0: return '🟢'
            elif value < 0: return '🔴'
            else: return '🟡'
        elif indicator == 'stoch':
            if value < 20: return '🟢'
            elif value > 80: return '🔴'
            else: return '🟡'
        else:
            return '📊'

    def _generate_enhanced_chart(self, prices: List[float], coin: str, indicators: Dict[str, Any], 
                               signal: str, confidence_level: Dict[str, Any], signal_config: Dict[str, Any]) -> str:
        try:
            plt.figure(figsize=(14, 8))
            
            # لون الخلفية بناء على نوع الإشارة
            background_color = 'lightgreen' if signal == 'BUY' else 'lightcoral' if signal == 'SELL' else 'lightblue'
            plt.gca().set_facecolor(background_color)
            
            # رسم السعر بلون مميز
            price_color = signal_config["chart_color"]
            plt.plot(prices[-50:], color=price_color, linewidth=3, label=f'السعر - {signal_config["title"]}', alpha=0.8)
            
            # رسم المتوسطات
            if len(prices) >= 21:
                ema_8 = pd.Series(prices).ewm(span=8, adjust=False).mean().values[-50:]
                ema_21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values[-50:]
                plt.plot(ema_8, color='orange', linewidth=2, label='EMA8', linestyle='--')
                plt.plot(ema_21, color='red', linewidth=2, label='EMA21', linestyle='-.')
            
            # إضافة علامة الإشارة
            signal_color = signal_config["color"]
            plt.axvline(x=len(prices[-50:])-1, color=signal_color, linestyle='-', alpha=0.9, linewidth=4, 
                       label=f'نقطة الإشارة - {signal_config["title"]}')
            
            # إضافة نقاط مهمة
            plt.scatter(len(prices[-50:])-1, prices[-1], color=signal_color, s=200, zorder=5, 
                       marker='*', edgecolors='white', linewidth=2)
            
            # العنوان المميز
            level_name = confidence_level["name"]
            signal_emoji = signal_config["signal_emoji"]
            plt.title(f"{signal_emoji} {coin.upper()} - إشارة {signal_config['title']} - ثقة {level_name} {signal_emoji}", 
                     fontsize=16, fontweight='bold', color=signal_color)
            
            plt.xlabel("الشموع (5 دقائق)", fontsize=12)
            plt.ylabel("السعر (USDT)", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor=background_color)
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

# باقي الكود يبقى كما هو (BinanceDataFetcher والمهام الأخرى)
class BinanceDataFetcher:
    """جلب بيانات من Binance للإطار 5 دقائق"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.analyzer = TradingSignalAnalyzer()
        self.cache = {}

    async def get_coin_data(self, coin_data: Dict[str, str]) -> Dict[str, Any]:
        cache_key = f"{coin_data['binance_symbol']}_5m"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < CACHE_TTL:
            return self.cache[cache_key]['data']
        
        try:
            data = await self._fetch_5m_data(coin_data['binance_symbol'])
            
            if not data.get('prices'):
                safe_log_error(f"فشل جلب بيانات 5m لـ {coin_data['symbol']}", coin_data['symbol'], "data_fetcher")
                return self._get_fallback_data(current_time)
            
            signal_analysis = self.analyzer.generate_trading_signals(data['prices'], data['volumes'])
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'signal_analysis': signal_analysis,
                'prices': data['prices'],
                'volumes': data['volumes'],
                'timestamp': current_time,
                'source': 'binance_5m'
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            safe_log_info(f"تم جلب بيانات 5m لـ {coin_data['symbol']} - الثقة: {signal_analysis['confidence']*100:.1f}%", coin_data['symbol'], "data_fetcher")
            return result
                
        except Exception as e:
            safe_log_error(f"خطأ في جلب بيانات {coin_data['symbol']}: {e}", coin_data['symbol'], "data_fetcher")
            return self._get_fallback_data(current_time)

    async def _fetch_5m_data(self, symbol: str) -> Dict[str, Any]:
        """جلب بيانات الإطار 5 دقائق من Binance"""
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=100"
        
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        return {
                            'prices': [float(item[4]) for item in data],
                            'highs': [float(item[2]) for item in data],
                            'lows': [float(item[3]) for item in data],
                            'volumes': [float(item[5]) for item in data],
                            'source': 'binance_5m'
                        }
                await asyncio.sleep(1 ** attempt)
            except Exception as e:
                safe_log_error(f"خطأ في جلب البيانات 5m: {e}", symbol, "binance_fetch")
                await asyncio.sleep(1 ** attempt)
        
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    def _get_fallback_data(self, timestamp: float) -> Dict[str, Any]:
        return {
            'price': 0,
            'signal_analysis': {
                "signal": "HOLD", 
                "confidence": 0, 
                "confidence_level": get_confidence_level(0),
                "action": "انتظار", 
                "signal_type": get_signal_type("HOLD"),
                "indicators": {}
            },
            'prices': [],
            'volumes': [],
            'timestamp': timestamp,
            'source': 'fallback'
        }

    async def close(self):
        await self.client.aclose()

# التهيئة والمهام الرئيسية
data_fetcher = BinanceDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def trading_signals_monitoring_task():
    safe_log_info(f"بدء مراقبة إشارات التداول - إطار 5 دقائق - عتبة الثقة: {CONFIDENCE_THRESHOLD*100}%", "all", "monitoring")
    
    while True:
        try:
            signals_sent = 0
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    data = await data_fetcher.get_coin_data(coin_data)
                    signal_analysis = data['signal_analysis']
                    confidence_level = signal_analysis['confidence_level']
                    signal_config = signal_analysis['signal_type']
                    
                    safe_log_info(f"{coin_key}: {signal_config['emoji']} {signal_analysis['signal']} (ثقة: {signal_analysis['confidence']*100:.1f}% - {confidence_level['name']})", coin_key, "monitoring")
                    
                    if confidence_level["send_alert"]:
                        success = await notifier.send_signal_alert(coin_key, signal_analysis, data['price'], data['prices'])
                        if success:
                            signals_sent += 1
                            safe_log_info(f"✅ تم إرسال إشعار إشارة {signal_config['emoji']} لـ {coin_key}", coin_key, "monitoring")
                            await asyncio.sleep(8)
                    else:
                        safe_log_info(f"⏭️ تخطي {coin_key}: مستوى الثقة {confidence_level['name']} لا يسمح بالإرسال", coin_key, "monitoring")
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    safe_log_error(f"خطأ في {coin_key}: {e}", coin_key, "monitoring")
                    continue
            
            safe_log_info(f"اكتملت دورة مراقبة الإشارات - تم إرسال {signals_sent} إشارة", "all", "monitoring")
            await asyncio.sleep(300)
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(60)

# endpoints تبقى كما هي
@app.get("/")
async def root():
    return {
        "message": "بوت إشارات التداول v5.0", 
        "version": "5.0.0", 
        "timeframe": "5 دقائق",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence_levels": CONFIDENCE_LEVELS,
        "signal_types": SIGNAL_TYPES,
        "data_source": "Binance مباشرة"
    }

@app.get("/signal/{coin}")
async def get_coin_signal(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "العملة غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    return {
        "coin": coin, 
        "price": data['price'], 
        "signal_analysis": data['signal_analysis'],
        "timeframe": "5m",
        "data_source": data['source']
    }

@app.get("/confidence-levels")
async def get_confidence_levels():
    return {
        "confidence_levels": CONFIDENCE_LEVELS,
        "current_threshold": CONFIDENCE_THRESHOLD
    }

@app.get("/status")
async def status():
    return {
        "status": "نشط - مراقبة إشارات التداول", 
        "version": "5.0.0",
        "timeframe": "5 دقائق",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "data_source": "Binance مباشرة",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "cache_size": len(data_fetcher.cache),
        "cache_ttl": CACHE_TTL
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info(f"بدء التشغيل - v5.0.0 - إشارات تداول 5 دقائق بتمييز واضح", "system", "startup")
    asyncio.create_task(trading_signals_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف بوت التداول", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
