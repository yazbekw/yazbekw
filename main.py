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

app = FastAPI(title="Crypto Trading Signals Bot", version="2.0.0")

# إعدادات البوت
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))  # 1 دقيقة لتحديث أسرع
CONFIDENCE_THRESHOLD = 0.30  # 65% عتبة ثقة أعلى

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

class TradingSignalAnalyzer:
    """محلل إشارات تداول دقيق"""
    
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
        
        # إعدادات مخصصة للإطار 5 دقائق
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
    def generate_trading_signals(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارات تداول دقيقة"""
        if len(prices) < 50:
            return {
                "signal": "HOLD", 
                "confidence": 0, 
                "action": "انتظار - بيانات غير كافية",
                "indicators": {}
            }
        
        try:
            current_price = prices[-1]
            
            # حساب المؤشرات
            rsi = TradingSignalAnalyzer.calculate_rsi(prices)
            macd_data = TradingSignalAnalyzer.calculate_macd(prices)
            ma_data = TradingSignalAnalyzer.calculate_moving_averages(prices)
            bb_data = TradingSignalAnalyzer.calculate_bollinger_bands(prices)
            
            # تحليل الحجم
            volume_trend = TradingSignalAnalyzer._analyze_volume_trend(volumes)
            
            # توليد الإشارات
            signal, confidence = TradingSignalAnalyzer._generate_signals(
                current_price, rsi, macd_data, ma_data, bb_data, volume_trend
            )
            
            # التوصية بناء على الإشارة
            action = TradingSignalAnalyzer._get_action_recommendation(signal, confidence, rsi)
            
            return {
                "signal": signal,
                "confidence": round(confidence, 2),
                "action": action,
                "indicators": {
                    "rsi": round(rsi, 1),
                    "macd_hist": macd_data['histogram'],
                    "macd_line": macd_data['macd'],
                    "macd_signal": macd_data['signal'],
                    "ema_8": ma_data['ema_8'],
                    "ema_21": ma_data['ema_21'],
                    "sma_20": ma_data['sma_20'],
                    "bb_upper": bb_data['upper'],
                    "bb_lower": bb_data['lower'],
                    "bb_position": "وسط" if bb_data['lower'] < current_price < bb_data['upper'] else "علوي" if current_price >= bb_data['upper'] else "سفلي",
                    "volume_trend": volume_trend,
                    "price_vs_ema8": "فوق" if current_price > ma_data['ema_8'] else "تحت",
                    "ema8_vs_ema21": "فوق" if ma_data['ema_8'] > ma_data['ema_21'] else "تحت"
                }
            }
            
        except Exception as e:
            safe_log_error(f"خطأ في توليد الإشارات: {e}", "N/A", "analyzer")
            return {"signal": "HOLD", "confidence": 0, "action": "انتظار", "indicators": {}}

    @staticmethod
    def _analyze_volume_trend(volumes: List[float]) -> str:
        """تحليل اتجاه الحجم"""
        if len(volumes) < 10:
            return "مستقر"
        
        recent_volume = np.mean(volumes[-5:])
        previous_volume = np.mean(volumes[-10:-5])
        
        if recent_volume > previous_volume * 1.2:
            return "متزايد"
        elif recent_volume < previous_volume * 0.8:
            return "متراجع"
        else:
            return "مستقر"

    @staticmethod
    def _generate_signals(current_price: float, rsi: float, macd_data: Dict[str, float], 
                         ma_data: Dict[str, float], bb_data: Dict[str, float], 
                         volume_trend: str) -> Tuple[str, float]:
        """توليد إشارات التداول الرئيسية"""
        
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # إشارة RSI
        if rsi < 35:
            buy_signals += 1
        elif rsi > 65:
            sell_signals += 1
        total_signals += 1
        
        # إشارة MACD
        if macd_data['histogram'] > 0 and macd_data['macd'] > macd_data['signal']:
            buy_signals += 1
        elif macd_data['histogram'] < 0 and macd_data['macd'] < macd_data['signal']:
            sell_signals += 1
        total_signals += 1
        
        # إشارة المتوسطات المتحركة
        if current_price > ma_data['ema_8'] and ma_data['ema_8'] > ma_data['ema_21']:
            buy_signals += 1
        elif current_price < ma_data['ema_8'] and ma_data['ema_8'] < ma_data['ema_21']:
            sell_signals += 1
        total_signals += 1
        
        # إشارة Bollinger Bands
        if current_price <= bb_data['lower']:
            buy_signals += 1
        elif current_price >= bb_data['upper']:
            sell_signals += 1
        total_signals += 1
        
        # إشارة الحجم
        if volume_trend == "متزايد":
            if buy_signals > sell_signals:
                buy_signals += 0.5
            elif sell_signals > buy_signals:
                sell_signals += 0.5
        total_signals += 0.5
        
        # تحديد الإشارة النهائية
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6 and buy_ratio > sell_ratio:
            confidence = min(buy_ratio * 0.9, 0.85)
            return "BUY", confidence
        elif sell_ratio > 0.6 and sell_ratio > buy_ratio:
            confidence = min(sell_ratio * 0.9, 0.85)
            return "SELL", confidence
        else:
            # HOLD مع ثقة منخفضة
            confidence = max(buy_ratio, sell_ratio) * 0.6
            return "HOLD", confidence

    @staticmethod
    def _get_action_recommendation(signal: str, confidence: float, rsi: float) -> str:
        """توصيات التداول بناء على الإشارة"""
        
        if signal == "BUY":
            if confidence > 0.75:
                if rsi < 70:
                    return "🟢 شراء قوي - دخول فوري"
                else:
                    return "🟡 شراء بحذر - RSI مرتفع"
            elif confidence > 0.65:
                return "🟢 إشارة شراء - انتظار تأكيد"
            else:
                return "⚪ مراقبة للشراء - ثقة منخفضة"
                
        elif signal == "SELL":
            if confidence > 0.75:
                if rsi > 30:
                    return "🔴 بيع قوي - خروج فوري"
                else:
                    return "🟠 بيع بحذر - RSI منخفض"
            elif confidence > 0.65:
                return "🔴 إشارة بيع - انتظار تأكيد"
            else:
                return "⚪ مراقبة للبيع - ثقة منخفضة"
                
        else:  # HOLD
            if confidence > 0.7:
                return "⚪ انتظار - السوق متجه للتجميع"
            elif confidence > 0.5:
                return "⚪ انتظار - اتجاه غير واضح"
            else:
                return "⚪ انتظار - إشارات متضاربة"

class TelegramNotifier:
    """إشعارات التداول"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        safe_log_info(f"تهيئة بعتبة ثقة: {self.confidence_threshold*100}%", "system", "config")

    async def send_signal_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        current_confidence = analysis["confidence"]
        signal_type = analysis["signal"]
        
        safe_log_info(f"فحص {coin}: {signal_type} بثقة {current_confidence*100}%", coin, "filter")
        
        if current_confidence < self.confidence_threshold:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: الثقة غير كافية", coin, "filter")
            return False
        
        # فحص واقعية المؤشرات
        rsi = analysis["indicators"]["rsi"]
        if (signal_type == "BUY" and rsi > 75) or (signal_type == "SELL" and rsi < 25):
            safe_log_info(f"🚫 تم رفض إشعار {coin}: RSI غير واقعي {rsi}", coin, "reality_check")
            return False
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: {signal_type} بثقة {current_confidence*100}%", coin, "filter")
        
        signal = analysis["signal"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        # إعداد الرسالة بناء على نوع الإشارة
        if signal == "BUY":
            emoji = "🟢"
            title = "إشارة شراء"
        elif signal == "SELL":
            emoji = "🔴" 
            title = "إشارة بيع"
        else:
            emoji = "⚪"
            title = "انتظار"
        
        message = f"{emoji} **{coin.upper()} - {title}**\n\n"
        message += f"💰 **السعر:** ${price:,.2f}\n"
        message += f"🎯 **الثقة:** {confidence*100:.1f}%\n"
        message += f"⚡ **التوصية:** {action}\n\n"
        
        message += f"🔍 **المؤشرات:**\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• MACD: {indicators['macd_hist']:.4f}\n"
        message += f"• السعر/EMA8: {indicators['price_vs_ema8']}\n"
        message += f"• EMA8/EMA21: {indicators['ema8_vs_ema21']}\n"
        message += f"• Bollinger Band: {indicators['bb_position']}\n"
        message += f"• الحجم: {indicators['volume_trend']}\n\n"
        
        message += f"📊 **المستويات:**\n"
        message += f"• EMA8: {indicators['ema_8']:.2f}\n"
        message += f"• EMA21: {indicators['ema_21']:.2f}\n"
        message += f"• BB علوي: {indicators['bb_upper']:.2f}\n"
        message += f"• BB سفلي: {indicators['bb_lower']:.2f}\n\n"
        
        message += f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ إطار 5 دقائق - v2.0"

        chart_base64 = self._generate_signal_chart(prices, coin, indicators, signal)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                safe_log_info(f"تم إرسال إشعار إشارة لـ {coin}", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _generate_signal_chart(self, prices: List[float], coin: str, indicators: Dict[str, Any], signal: str) -> str:
        try:
            plt.figure(figsize=(10, 6))
            
            # رسم السعر
            plt.plot(prices[-50:], color='blue', linewidth=2, label='السعر')
            
            # رسم المتوسطات إذا كانت متوفرة
            if len(prices) >= 21:
                ema_8 = pd.Series(prices).ewm(span=8, adjust=False).mean().values[-50:]
                ema_21 = pd.Series(prices).ewm(span=21, adjust=False).mean().values[-50:]
                plt.plot(ema_8, color='orange', linewidth=1.5, label='EMA(8)')
                plt.plot(ema_21, color='red', linewidth=1.5, label='EMA(21)')
            
            # إضافة علامة الإشارة
            color = 'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'gray'
            plt.axvline(x=len(prices[-50:])-1, color=color, linestyle='--', alpha=0.7, label=f'إشارة {signal}')
            
            plt.title(f"{coin.upper()} - إشارة {signal} - إطار 5 دقائق")
            plt.xlabel("الشموع")
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
            # جلب بيانات 5 دقائق من Binance
            data = await self._fetch_5m_data(coin_data['binance_symbol'])
            
            if not data.get('prices'):
                safe_log_error(f"فشل جلب بيانات 5m لـ {coin_data['symbol']}", coin_data['symbol'], "data_fetcher")
                return self._get_fallback_data(current_time)
            
            signal_analysis = self.analyzer.generate_trading_signals(
                data['prices'], data['volumes']
            )
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'signal_analysis': signal_analysis,
                'prices': data['prices'],
                'volumes': data['volumes'],
                'timestamp': current_time,
                'source': 'binance_5m'
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            safe_log_info(f"تم جلب بيانات 5m لـ {coin_data['symbol']}", coin_data['symbol'], "data_fetcher")
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
                            'prices': [float(item[4]) for item in data],  # سعر الإغلاق
                            'highs': [float(item[2]) for item in data],   # أعلى سعر
                            'lows': [float(item[3]) for item in data],    # أدنى سعر
                            'volumes': [float(item[5]) for item in data], # حجم التداول
                            'source': 'binance_5m'
                        }
                await asyncio.sleep(1 ** attempt)
            except Exception as e:
                safe_log_error(f"خطأ في جلب البيانات 5m: {e}", symbol, "binance_fetch")
                await asyncio.sleep(1 ** attempt)
        
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    def _get_fallback_data(self, timestamp: float) -> Dict[str, Any]:
        """بيانات احتياطية عند الفشل"""
        return {
            'price': 0,
            'signal_analysis': {"signal": "HOLD", "confidence": 0, "action": "انتظار", "indicators": {}},
            'prices': [],
            'volumes': [],
            'timestamp': timestamp,
            'source': 'fallback'
        }

    async def close(self):
        await self.client.aclose()

# التهيئة
data_fetcher = BinanceDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def trading_signals_monitoring_task():
    """مراقبة إشارات التداول كل 5 دقائق"""
    safe_log_info(f"بدء مراقبة إشارات التداول - إطار 5 دقائق", "all", "monitoring")
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    data = await data_fetcher.get_coin_data(coin_data)
                    signal_analysis = data['signal_analysis']
                    
                    safe_log_info(f"{coin_key}: {signal_analysis['signal']} (ثقة: {signal_analysis['confidence']*100:.1f}%)", coin_key, "monitoring")
                    
                    if signal_analysis['confidence'] >= CONFIDENCE_THRESHOLD and signal_analysis['signal'] != "HOLD":
                        success = await notifier.send_signal_alert(coin_key, signal_analysis, data['price'], data['prices'])
                        if success:
                            safe_log_info(f"✅ تم إرسال إشعار إشارة لـ {coin_key}", coin_key, "monitoring")
                            # انتظار بعد إرسال إشعار ناجح لتجنب التكرار
                            await asyncio.sleep(10)
                    else:
                        safe_log_info(f"⏭️ تخطي {coin_key}: {signal_analysis['signal']} بثقة غير كافية", coin_key, "monitoring")
                    
                    await asyncio.sleep(1)  # انتظار قصير بين العملات
                    
                except Exception as e:
                    safe_log_error(f"خطأ في {coin_key}: {e}", coin_key, "monitoring")
                    continue
                    
            safe_log_info("اكتملت دورة مراقبة الإشارات", "all", "monitoring")
            await asyncio.sleep(300)  # 5 دقائق بين الدورات
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(60)

@app.get("/")
async def root():
    return {
        "message": "بوت إشارات التداول v2.0", 
        "version": "2.0.0", 
        "timeframe": "5 دقائق",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
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

@app.get("/status")
async def status():
    return {
        "status": "نشط - مراقبة إشارات التداول", 
        "version": "2.0.0",
        "timeframe": "5 دقائق",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "data_source": "Binance مباشرة",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "cache_size": len(data_fetcher.cache),
        "cache_ttl": CACHE_TTL
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info(f"بدء التشغيل - v2.0.0 - إشارات تداول 5 دقائق", "system", "startup")
    asyncio.create_task(trading_signals_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف بوت التداول", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
