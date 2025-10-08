from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import math
from datetime import datetime
import logging
from typing import Dict, Any, List
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from logging.handlers import RotatingFileHandler
from scipy.signal import find_peaks

# إعداد التسجيل (Structured Logging + File Rotation) - معدل
logger = logging.getLogger("crypto_bot")
logger.setLevel(logging.INFO)

# إزالة أي معالجات موجودة مسبقاً
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler with rotation (max 5MB, keep 3 backups)
try:
    file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"تعذر إنشاء ملف التسجيل: {e}")

# منع double logging
logger.propagate = False
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

app = FastAPI(title="Crypto Market Phase Bot", version="8.5.0")  # تحديث الإصدار

# إعدادات التلغرام والبيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))  # 15 دقيقة
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))  # ⭐ رفع العتبة إلى 85%

# تعريف العملات مع تخصيص المؤشرات
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC',
            'volatility_threshold': 0.04, 'rsi_low': 55, 'rsi_high': 65},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH',
            'volatility_threshold': 0.06, 'rsi_low': 50, 'rsi_high': 70},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB',
            'volatility_threshold': 0.05, 'rsi_low': 50, 'rsi_high': 70},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL',
            'volatility_threshold': 0.07, 'rsi_low': 45, 'rsi_high': 75},
    'ada': {'name': 'Cardano', 'coingecko_id': 'cardano', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA',
            'volatility_threshold': 0.05, 'rsi_low': 50, 'rsi_high': 70},
    'xrp': {'name': 'XRP', 'coingecko_id': 'ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP',
            'volatility_threshold': 0.06, 'rsi_low': 50, 'rsi_high': 70},
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT',
            'volatility_threshold': 0.05, 'rsi_low': 50, 'rsi_high': 70}
}

def safe_log_info(message: str, coin: str = "system", source: str = "app"):
    """تسجيل آمن بدون معلمات إضافية"""
    try:
        logger.info(f"{message} - Coin: {coin} - Source: {source}")
    except Exception as e:
        print(f"خطأ في التسجيل: {e} - الرسالة: {message}")

def safe_log_error(message: str, coin: str = "system", source: str = "app"):
    """تسجيل أخطاء آمن"""
    try:
        logger.error(f"{message} - Coin: {coin} - Source: {source}")
    except Exception as e:
        print(f"خطأ في تسجيل الخطأ: {e} - الرسالة: {message}")

def validate_analysis_result(analysis: Dict[str, Any]) -> bool:
    """التحقق من صحة نتيجة التحليل"""
    required_keys = ['phase', 'confidence', 'action']
    if not all(key in analysis for key in required_keys):
        return False
    
    if not isinstance(analysis['confidence'], (int, float)):
        return False
    
    if not 0 <= analysis['confidence'] <= 1:
        return False
    
    return True

class MarketPhaseAnalyzer:
    """محلل مراحل السوق بناءً على نظرية وايكوف مع نظريات إضافية"""
    
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], sentiment_score: float, coin_custom: Dict) -> Dict[str, Any]:
        """تحليل مرحلة السوق الحالية مع دمج نظريات إضافية"""
        if len(prices) < 50:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار"}
        
        try:
            df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
            
            # المؤشرات الأساسية (مع تخصيص per-coin)
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # الحجم النسبي
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # تقلبات السعر (مع تخصيص)
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            
            # ATR
            df['tr'] = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            # VSA
            df['spread'] = df['high'] - df['low']
            df['spread_volume_ratio'] = df['spread'] / df['volume'].replace(0, 1e-10)
            spread_volume_mean = df['spread_volume_ratio'].mean()
            
            # Ichimoku Cloud
            df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
            
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            # تحسين Elliott Wave Detection مع فيبوناتشي
            elliott_wave = MarketPhaseAnalyzer._detect_elliott_waves(prices, highs, lows)
            
            phase_analysis = MarketPhaseAnalyzer._determine_phase(latest, prev, sentiment_score, elliott_wave, spread_volume_mean, coin_custom)
            
            # التحقق من صحة النتيجة
            if not validate_analysis_result(phase_analysis):
                return {"phase": "غير محدد", "confidence": 0, "action": "انتظار"}
            
            return phase_analysis
            
        except Exception as e:
            safe_log_error(f"خطأ في تحليل المرحلة: {e}", "N/A", "analyzer")
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار"}
    
    @staticmethod
    def _detect_elliott_waves(prices: List[float], highs: List[float], lows: List[float]) -> str:
        """كشف موجات إليوت محسن مع التحقق من نسب فيبوناتشي"""
        if len(prices) < 30:
            return "غير محدد - بيانات غير كافية"
        
        try:
            # كشف القمم والقيعان بإعدادات محسنة
            peaks, _ = find_peaks(highs, distance=5, prominence=0.02)
            troughs, _ = find_peaks([-l for l in lows], distance=5, prominence=0.02)
            
            if len(peaks) < 2 or len(troughs) < 2:
                return "غير محدد - قمم وقيعان غير كافية"
            
            # حساب retracements وتحقق فيبوناتشي
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            waves = []
            
            for i in range(min(len(peaks), len(troughs)) - 1):
                if i + 1 < len(peaks) and i < len(troughs):
                    wave_up = highs[peaks[i+1]] - lows[troughs[i]]
                    wave_down = highs[peaks[i]] - lows[troughs[i+1]]
                    retrace = abs(wave_down / wave_up) if wave_up != 0 else 0
                    
                    if any(abs(retrace - fib) < 0.05 for fib in fib_ratios):
                        waves.append("موجة صعودية مع تصحيح فيبوناتشي")
                    else:
                        waves.append("موجة تصحيحية")
            
            # تحليل الاتجاه الأساسي
            if prices[-1] > prices[0]:
                base_trend = "اتجاه صاعد محتمل"
            else:
                base_trend = "اتجاه هابط محتمل"
            
            if waves and "صعودية" in waves[-1]:
                return f"صعودية محتملة مع فيبوناتشي - {base_trend}"
            elif waves:
                return f"تصحيحية محتملة - {base_trend}"
            else:
                return base_trend
                
        except Exception as e:
            return f"خطأ في التحليل: {str(e)}"
    
    @staticmethod
    def _determine_phase(latest, prev, sentiment_score: float, elliott_wave: str, spread_volume_mean: float, coin_custom: Dict) -> Dict[str, Any]:
        """تحديد المرحلة مع تخصيص per-coin"""
        vol_thresh = coin_custom.get('volatility_threshold', 0.05)
        rsi_low = coin_custom.get('rsi_low', 60)
        rsi_high = coin_custom.get('rsi_high', 70)
        
        accumulation_signs = [
            latest['volatility'] < vol_thresh,
            latest['volume_ratio'] < 1.2,
            latest['rsi'] < rsi_low,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.05,
            latest['macd_hist'] > 0,
            latest['close'] > latest['bb_lower'],
            latest['atr'] / latest['close'] < 0.03,
            latest['spread_volume_ratio'] < spread_volume_mean,
            latest['close'] > latest['senkou_span_a'] and latest['close'] > latest['senkou_span_b'],
            sentiment_score < 0.5,
            "تصحيحية" in elliott_wave or "هابط" in elliott_wave
        ]
        
        markup_signs = [
            latest['close'] > latest['sma20'] > latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] > 50,
            latest['close'] > prev['close'],
            latest['macd'] > latest['macd_signal'],
            latest['close'] > latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.02,
            latest['spread_volume_ratio'] > spread_volume_mean,
            latest['tenkan_sen'] > latest['kijun_sen'],
            sentiment_score > 0.6,
            "صعودية" in elliott_wave or "صاعد" in elliott_wave
        ]
        
        distribution_signs = [
            latest['volatility'] > vol_thresh + 0.03,
            latest['volume_ratio'] > 1.5,
            latest['rsi'] > rsi_high,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.1,
            latest['macd_hist'] < 0,
            latest['close'] < latest['bb_upper'],
            latest['atr'] / latest['close'] > 0.04,
            latest['spread_volume_ratio'] < spread_volume_mean,
            latest['close'] < latest['senkou_span_a'] or latest['close'] < latest['senkou_span_b'],
            sentiment_score > 0.8,
            "تصحيحية" in elliott_wave
        ]
        
        markdown_signs = [
            latest['close'] < latest['sma20'] < latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] < 40,
            latest['close'] < prev['close'],
            latest['macd'] < latest['macd_signal'],
            latest['close'] < latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.03,
            latest['spread_volume_ratio'] > spread_volume_mean,
            latest['tenkan_sen'] < latest['kijun_sen'],
            sentiment_score < 0.4,
            "تصحيحية" in elliott_wave or "هابط" in elliott_wave
        ]
        
        scores = {
            "تجميع": sum(accumulation_signs),
            "صعود": sum(markup_signs),
            "توزيع": sum(distribution_signs),
            "هبوط": sum(markdown_signs)
        }
        
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / 11.0
        
        action = MarketPhaseAnalyzer._get_action_recommendation(best_phase, confidence, latest)
        
        return {
            "phase": best_phase,
            "confidence": round(confidence, 2),
            "action": action,
            "scores": scores,
            "indicators": {
                "rsi": round(latest['rsi'], 1) if not pd.isna(latest['rsi']) else 50,
                "volume_ratio": round(latest['volume_ratio'], 2) if not pd.isna(latest['volume_ratio']) else 1.0,
                "volatility": round(latest['volatility'], 3) if not pd.isna(latest['volatility']) else 0.0,
                "macd_hist": round(latest['macd_hist'], 3) if not pd.isna(latest['macd_hist']) else 0.0,
                "bb_position": round((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']), 2) if not pd.isna(latest['bb_lower']) and not pd.isna(latest['bb_upper']) else 0.5,
                "atr_ratio": round(latest['atr'] / latest['close'], 3) if not pd.isna(latest['atr']) else 0.0,
                "spread_volume_ratio": round(latest['spread_volume_ratio'], 3) if not pd.isna(latest['spread_volume_ratio']) else 0.0,
                "ichimoku_trend": "صاعد" if latest['close'] > latest['senkou_span_a'] else "هابط",
                "sentiment_score": round(sentiment_score, 2),
                "elliott_wave": elliott_wave,
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, latest) -> str:
        actions = {
            "تجميع": "مراقبة للشراء عند الكسر. دعم محتمل عند مستوى ATR السفلي.",
            "صعود": "شراء على الارتدادات. هدف محتمل عند مستوى Fibonacci 1.618.",
            "توزيع": "استعداد للبيع. مقاومة محتملة عند مستوى المقاومة النفسية.",
            "هبوط": "بيع على الارتدادات. هدف محتمل عند مستوى الدعم القوي."
        }
        base_action = actions.get(phase, "انتظار بسبب عدم وضوح الإشارات.")
        
        if confidence > CONFIDENCE_THRESHOLD:
            if phase == "تجميع":
                return f"🟢 استعداد للشراء - مرحلة تجميع قوية. {base_action} (ثقة عالية: {confidence*100}%)"
            elif phase == "صعود":
                return f"🟢 شراء - اتجاه صاعد قوي. {base_action}"
            elif phase == "توزيع":
                return f"🔴 بيع - مرحلة توزيع نشطة. {base_action}"
            elif phase == "هبوط":
                return f"🔴 بيع - اتجاه هابط قوي. {base_action}"
        
        return base_action

class TelegramNotifier:
    """إشعارات تلغرام محسنة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification_time = {}
        self.min_notification_interval = 0
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        """إرسال إشعار مرحلة السوق مع تصفية صارمة للثقة"""
        current_confidence = analysis["confidence"]
        
        # 🔴 التحقق الصارم من عتبة الثقة
        if current_confidence < self.confidence_threshold:
            safe_log_info(f"🚫 تم رفض إشعار {coin}: الثقة {current_confidence*100}% < العتبة {self.confidence_threshold*100}%", coin, "telegram_filter")
            return False
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: الثقة {current_confidence*100}% >= العتبة {self.confidence_threshold*100}%", coin, "telegram_filter")
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        # 🆕 تحديث تنسيق الإشعار للتأكد من التعديل
        message = f"🚨 **{coin.upper()} - إشعار مرحلة {phase}** 🚨\n"
        message += f"💰 السعر الحالي: ${price:,.2f}\n"
        message += f"📊 مستوى الثقة: {confidence*100}%\n"
        message += f"🎯 التوصية: {action}\n\n"
        
        message += f"📈 المؤشرات الفنية:\n"
        message += f"• 📊 RSI: {indicators['rsi']}\n"
        message += f"• 💰 الحجم النسبي: {indicators['volume_ratio']}x\n"
        message += f"• 📉 التقلب: {indicators['volatility']*100}%\n"
        message += f"• 🔄 MACD: {indicators['macd_hist']}\n"
        message += f"• 📊 بولينجر: {indicators['bb_position']*100}%\n"
        message += f"• 📈 VSA: {indicators['spread_volume_ratio']}\n"
        message += f"• ☁️ إيشيموكو: {indicators['ichimoku_trend']}\n"
        message += f"• 🌊 موجات إليوت: {indicators['elliott_wave']}\n"
        message += f"• 📈 الاتجاه العام: {indicators['trend']}\n\n"
        
        message += f"🕒 وقت التحليل: {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ الإصدار: 8.5.0 (محدث)\n"
        message += f"⚠️ تحذير: هذا ليس نصيحة استثمارية.\n"
        message += f"🔒 مرشح الثقة: {self.confidence_threshold*100}%"
        
        chart_base64 = self._generate_price_chart(prices, coin)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                self.last_notification_time[f"{coin}_phase"] = time.time()
                safe_log_info(f"تم إرسال إشعار لـ {coin} بنجاح (ثقة: {confidence*100}%)", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _generate_price_chart(self, prices: List[float], coin: str) -> str:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(prices, label=f"{coin.upper()} السعر", color='blue', linewidth=2)
            plt.title(f"اتجاه سعر {coin.upper()} - الإصدار 8.5.0", fontsize=14)
            plt.xlabel("الفترة الزمنية")
            plt.ylabel("السعر (USD)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            safe_log_error(f"خطأ في إنشاء الرسم البياني: {e}", coin, "chart")
            return ""

    async def send_simple_analysis(self, coin: str, price: float, phase: str, signal: str):
        if price == 0:
            safe_log_info(f"تم تخطي إشعار بسيط لـ {coin} بسبب سعر 0", coin, "telegram")
            return False
        
        message = f"💰 **{coin.upper()} تحديث سريع**\n"
        message += f"💵 السعر: ${price:,.2f}\n"
        message += f"📊 المرحلة: {phase}\n"
        message += f"🎯 الإشارة: {signal}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')}\n"
        message += f"⚡ الإصدار: 8.5.0"
        
        return await self._send_message(message)

    async def _send_message(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            return False
            
        try:
            if len(message) > 4096:
                message = message[:4090] + "..."
                
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", json=payload, timeout=15.0)
                
            return response.status_code == 200
        except Exception as e:
            safe_log_error(f"خطأ في إرسال الرسالة: {e}", "system", "telegram")
            return False

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
    """جلب بيانات العملات مع تحسين التوحيد"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.phase_analyzer = MarketPhaseAnalyzer()
        self.cache = {}
        self.rate_limit_remaining = {'coingecko': 50, 'binance': 1200}
        self.rate_limit_reset = {'coingecko': 0, 'binance': 0}

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
            
            # تحليل المشاعر باستخدام بيانات السوق (بدون تويتر)
            sentiment_score = await self._get_sentiment_from_market(data['prices'])
            
            phase_analysis = self.phase_analyzer.analyze_market_phase(
                data['prices'], data['highs'], data['lows'], data['volumes'], sentiment_score, coin_data
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
            await notifier.send_simple_analysis(coin_data['symbol'], 0, "غير محدد", f"فشل: {str(e)}")
            return {'price': 0, 'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار"}, 'prices': [], 'timestamp': current_time, 'source': 'fallback'}

    async def _fetch_from_coingecko(self, coin_id: str) -> Dict[str, Any]:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=100&interval=daily"
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    prices = [item[1] for item in data.get('prices', [])][-100:]
                    volumes = [item[1] for item in data.get('total_volumes', [])][-100:]
                    highs = [p * 1.01 for p in prices]  # تقدير للقيم العالية
                    lows = [p * 0.99 for p in prices]   # تقدير للقيم المنخفضة
                    return {'prices': prices, 'highs': highs, 'lows': lows, 'volumes': volumes, 'source': 'coingecko'}
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                safe_log_info(f"محاولة {attempt+1} فشلت لـ Coingecko: {e}", coin_id, "coingecko")
                await asyncio.sleep(2 ** attempt)
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'coingecko_failed'}

    async def _fetch_from_binance(self, symbol: str) -> Dict[str, Any]:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=100"
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'prices': [float(item[4]) for item in data],  # سعر الإغلاق
                        'highs': [float(item[2]) for item in data],   # أعلى سعر
                        'lows': [float(item[3]) for item in data],    # أدنى سعر
                        'volumes': [float(item[5]) for item in data], # حجم التداول
                        'source': 'binance'
                    }
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                safe_log_info(f"محاولة {attempt+1} فشلت لـ Binance: {e}", symbol, "binance")
                await asyncio.sleep(2 ** attempt)
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    async def _get_sentiment_from_market(self, prices: List[float]) -> float:
        """تحليل المشاعر باستخدام بيانات السوق فقط (بدون تويتر)"""
        try:
            if len(prices) < 10:
                return 0.5  # قيمة محايدة
            
            # استخدام تقلبات السوق كمؤشر للمشاعر
            recent_prices = prices[-10:]  # آخر 10 أيام
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # تحويل تغير السعر إلى مؤشر مشاعر (0-1)
            # تغير +10% => مشاعر 0.7، تغير -10% => مشاعر 0.3
            sentiment = 0.5 + (price_change * 2)
            
            # التأكد من أن القيمة بين 0 و 1
            sentiment = max(0.0, min(1.0, sentiment))
            
            safe_log_info(f"تحليل المشاعر: تغير السعر {price_change:.2%} => مشاعر {sentiment:.2f}", "system", "sentiment")
            return sentiment
            
        except Exception as e:
            safe_log_error(f"خطأ في تحليل المشاعر: {e}، استخدام قيمة محايدة", "system", "sentiment")
            return 0.5  # قيمة محايدة في حالة الخطأ

    def _update_rate_limits(self, headers, source: str):
        if 'x-ratelimit-remaining' in headers:
            self.rate_limit_remaining[source] = int(headers['x-ratelimit-remaining'])
        if 'x-ratelimit-reset' in headers:
            self.rate_limit_reset[source] = int(headers['x-ratelimit-reset'])

    async def close(self):
        await self.client.aclose()

# تهيئة
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def market_monitoring_task():
    """مهمة المراقبة الرئيسية مع معالجة محسنة للأخطاء"""
    safe_log_info("بدء مهمة مراقبة السوق", "all", "monitoring")
    safe_log_info(f"عتبة الثقة المطلوبة: {CONFIDENCE_THRESHOLD*100}%", "all", "monitoring")
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    safe_log_info(f"معالجة {coin_key}", coin_key, "monitoring")
                    
                    data = await data_fetcher.get_coin_data(coin_data)
                    phase_analysis = data['phase_analysis']
                    
                    safe_log_info(f"نتيجة تحليل {coin_key}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']*100}%)", 
                                 coin_key, "monitoring")
                    
                    # 🔴 تصفية صارمة بناءً على عتبة الثقة
                    if phase_analysis['confidence'] >= CONFIDENCE_THRESHOLD:  # استخدام >= بدلاً من >
                        success = await notifier.send_phase_alert(coin_key, phase_analysis, data['price'], data['prices'])
                        if success:
                            safe_log_info(f"تم إرسال إشعار لـ {coin_key} بنجاح", coin_key, "monitoring")
                        else:
                            safe_log_error(f"فشل إرسال إشعار لـ {coin_key}", coin_key, "monitoring")
                    else:
                        safe_log_info(f"تم تخطي إشعار {coin_key}: الثقة {phase_analysis['confidence']*100}% < العتبة {CONFIDENCE_THRESHOLD*100}%", 
                                     coin_key, "monitoring_filter")
                    
                    await asyncio.sleep(5)  # ⏱️ تقليل الفاصل بين العملات
                    
                except Exception as e:
                    safe_log_error(f"خطأ في معالجة {coin_key}: {e}", coin_key, "monitoring")
                    continue
                    
            safe_log_info("اكتملت دورة المراقبة، انتظار 5 دقائق", "all", "monitoring")
            await asyncio.sleep(300)  # ⏱️ 5 دقائق بين الدورات الكاملة
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(60)  # انتظار دقيقة قبل إعادة المحاولة

@app.get("/")
async def root():
    return {"message": "بوت محسن لتحسين الدقة", "version": "8.5.0", "confidence_threshold": CONFIDENCE_THRESHOLD}

@app.get("/phase/{coin}")
async def get_coin_phase(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    return {"coin": coin, "price": data['price'], "phase_analysis": data['phase_analysis']}

@app.get("/alert/{coin}")
async def send_phase_alert(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    success = await notifier.send_phase_alert(coin, data['phase_analysis'], data['price'], data['prices'])
    return {"success": success, "phase": data['phase_analysis']['phase']}

@app.get("/status")
async def status():
    return {
        "status": "نشط", 
        "supported_coins": list(SUPPORTED_COINS.keys()), 
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "cache_size": len(data_fetcher.cache),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """فحص صحة التطبيق"""
    try:
        # فحص الاتصال بالإنترنت وAPIs
        async with httpx.AsyncClient() as client:
            binance_response = await client.get("https://api.binance.com/api/v3/ping", timeout=10.0)
            coingecko_response = await client.get("https://api.coingecko.com/api/v3/ping", timeout=10.0)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cache_size": len(data_fetcher.cache),
            "binance_connection": binance_response.status_code == 200,
            "coingecko_connection": coingecko_response.status_code == 200,
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "monitoring_active": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    safe_log_info("بدء تشغيل البوت", "all", "system")
    safe_log_info(f"عتبة الثقة: {CONFIDENCE_THRESHOLD*100}%", "all", "system")
    asyncio.create_task(market_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف البوت", "all", "system")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
