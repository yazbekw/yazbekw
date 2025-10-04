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

# إعداد التسجيل (Structured Logging + File Rotation)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler with rotation (max 5MB, keep 3 backups)
file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s - Coin: %(coin)s - Source: %(source)s'
))
logger.addHandler(file_handler)

app = FastAPI(title="Crypto Market Phase Bot", version="8.1.0")

# إعدادات التلغرام والبيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))  # 15 دقيقة لتجنب إرهاق API
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))  # ⭐ عتبة الثقة من متغير البيئة

# تعريف العملات
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL'},
    'ada': {'name': 'Cardano', 'coingecko_id': 'cardano', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA'},
    'xrp': {'name': 'XRP', 'coingecko_id': 'ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP'},
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT'}
}

class MarketPhaseAnalyzer:
    """محلل مراحل السوق بناءً على نظرية وايكوف مع نظريات إضافية"""
    
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], sentiment_score: float) -> Dict[str, Any]:
        """تحليل مرحلة السوق الحالية مع دمج نظريات إضافية"""
        if len(prices) < 50:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار"}
        
        try:
            df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
            
            # المؤشرات الأساسية
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
            
            # تقلبات السعر
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
            
            # ATR (Average True Range)
            df['tr'] = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            # إضافة VSA (Volume Spread Analysis)
            df['spread'] = df['high'] - df['low']
            df['spread_volume_ratio'] = df['spread'] / df['volume'].replace(0, 1e-10)
            spread_volume_mean = df['spread_volume_ratio'].mean()  # حساب المتوسط هنا
            
            # إضافة Ichimoku Cloud
            df['tenkan_sen'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            df['kijun_sen'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            df['senkou_span_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
            
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            # إضافة Elliott Wave Detection
            elliott_wave = MarketPhaseAnalyzer._detect_elliott_waves(prices)
            
            phase_analysis = MarketPhaseAnalyzer._determine_phase(latest, prev, sentiment_score, elliott_wave, spread_volume_mean)
            return phase_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المرحلة: {e}", extra={"coin": "N/A", "source": "N/A"})
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار"}
    
    @staticmethod
    def _detect_elliott_waves(prices: List[float]) -> str:
        """كشف موجات إليوت البسيط"""
        peaks, _ = find_peaks(prices, distance=10)
        troughs, _ = find_peaks([-p for p in prices], distance=10)
        if len(peaks) >= 3 and len(troughs) >= 2:
            return "موجة صعودية محتملة" if prices[-1] > prices[peaks[-1]] else "موجة تصحيحية محتملة"
        return "غير محدد"
    
    @staticmethod
    def _determine_phase(latest, prev, sentiment_score: float, elliott_wave: str, spread_volume_mean: float) -> Dict[str, Any]:
        """تحديد المرحلة بناءً على المؤشرات الموسعة بما في ذلك النظريات الجديدة"""
        accumulation_signs = [
            latest['volatility'] < 0.05,
            latest['volume_ratio'] < 1.2,
            latest['rsi'] < 60,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.05,
            latest['macd_hist'] > 0,
            latest['close'] > latest['bb_lower'],
            latest['atr'] / latest['close'] < 0.03,
            latest['spread_volume_ratio'] < spread_volume_mean,  # VSA
            latest['close'] > latest['senkou_span_a'] and latest['close'] > latest['senkou_span_b'],  # Ichimoku
            sentiment_score < 0.5,  # Sentiment منخفض يشير إلى تجميع
            "تصحيحية" in elliott_wave  # Elliott Wave
        ]
        
        markup_signs = [
            latest['close'] > latest['sma20'] > latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] > 50,
            latest['close'] > prev['close'],
            latest['macd'] > latest['macd_signal'],
            latest['close'] > latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.02,
            latest['spread_volume_ratio'] > spread_volume_mean,  # VSA
            latest['tenkan_sen'] > latest['kijun_sen'],  # Ichimoku
            sentiment_score > 0.6,  # Sentiment إيجابي
            "صعودية" in elliott_wave  # Elliott Wave
        ]
        
        distribution_signs = [
            latest['volatility'] > 0.08,
            latest['volume_ratio'] > 1.5,
            latest['rsi'] > 70,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.1,
            latest['macd_hist'] < 0,
            latest['close'] < latest['bb_upper'],
            latest['atr'] / latest['close'] > 0.04,
            latest['spread_volume_ratio'] < spread_volume_mean,  # VSA (حجم مرتفع مع spread صغير)
            latest['close'] < latest['senkou_span_a'] or latest['close'] < latest['senkou_span_b'],  # Ichimoku
            sentiment_score > 0.8,  # Sentiment ذروة إيجابية
            "تصحيحية" in elliott_wave  # Elliott Wave
        ]
        
        markdown_signs = [
            latest['close'] < latest['sma20'] < latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] < 40,
            latest['close'] < prev['close'],
            latest['macd'] < latest['macd_signal'],
            latest['close'] < latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.03,
            latest['spread_volume_ratio'] > spread_volume_mean,  # VSA
            latest['tenkan_sen'] < latest['kijun_sen'],  # Ichimoku
            sentiment_score < 0.4,  # Sentiment سلبي
            "تصحيحية" in elliott_wave  # Elliott Wave
        ]
        
        scores = {
            "تجميع": sum(accumulation_signs),
            "صعود": sum(markup_signs),
            "توزيع": sum(distribution_signs),
            "هبوط": sum(markdown_signs)
        }
        
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / 11.0  # معدلة لـ 11 علامة (مع النظريات الجديدة)
        
        action = MarketPhaseAnalyzer._get_action_recommendation(best_phase, confidence, latest)
        
        return {
            "phase": best_phase,
            "confidence": round(confidence, 2),
            "action": action,
            "scores": scores,
            "indicators": {
                "rsi": round(latest['rsi'], 1),
                "volume_ratio": round(latest['volume_ratio'], 2),
                "volatility": round(latest['volatility'], 3),
                "macd_hist": round(latest['macd_hist'], 3),
                "bb_position": round((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']), 2),
                "atr_ratio": round(latest['atr'] / latest['close'], 3),
                "spread_volume_ratio": round(latest['spread_volume_ratio'], 3),
                "ichimoku_trend": "صاعد" if latest['close'] > latest['senkou_span_a'] else "هابط",
                "sentiment_score": round(sentiment_score, 2),
                "elliott_wave": elliott_wave,
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, latest) -> str:
        """تحديد الإجراء المناسب للمرحلة مع دعم قرار احترافي"""
        actions = {
            "تجميع": "مراقبة للشراء عند الكسر. دعم محتمل عند مستوى ATR السفلي.",
            "صعود": "شراء على الارتدادات. هدف محتمل عند مستوى Ichimoku العلوي.",
            "توزيع": "استعداد للبيع. مقاومة محتملة عند BB العلوي.",
            "هبوط": "بيع على الارتدادات. هدف محتمل عند مستوى ATR السفلي."
        }
        base_action = actions.get(phase, "انتظار بسبب عدم وضوح الإشارات.")
        
        # ⭐ استخدام عتبة الثقة من متغير البيئة
        if confidence > CONFIDENCE_THRESHOLD:
            if phase == "تجميع":
                return f"استعداد للشراء - مرحلة تجميع قوية. دعم القرار: {base_action} (ثقة عالية بناءً على VSA وElliott Waves)."
            elif phase == "صعود":
                return f"شراء - اتجاه صاعد قوي. دعم القرار: {base_action} (مدعوم بـ Ichimoku وSentiment إيجابي)."
            elif phase == "توزيع":
                return f"بيع - مرحلة توزيع نشطة. دعم القرار: {base_action} (تحذير من ذروة المشاعر)."
            elif phase == "هبوط":
                return f"بيع - اتجاه هابط قوي. دعم القرار: {base_action} (مدعوم بـ VSA وموجات تصحيحية)."
        
        return base_action

class TelegramNotifier:
    """إشعارات تلغرام محسنة مع إشعارات احترافية قوية"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification_time = {}
        self.min_notification_interval = 0  # لا انتظار بين الإشعارات
        # ⭐ استخدام عتبة الثقة من متغير البيئة
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        current_time = time.time()
        coin_key = f"{coin}_phase"
        
        # ⭐ استخدام عتبة الثقة من متغير البيئة
        if analysis["confidence"] < self.confidence_threshold:
            logger.info(f"تم تخطي إشعار {coin}: الثقة منخفضة ({analysis['confidence']})", 
                        extra={"coin": coin, "source": "telegram"})
            return False
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        # إنشاء رسالة احترافية قوية مع دعم قرار
        message = f"🎯 **{coin.upper()} - مرحلة {phase} (ثقة عالية)**\n"
        message += f"💰 السعر الحالي: ${price:,.2f}\n"
        message += f"📊 مستوى الثقة: {confidence*100}%\n"
        message += f"⚡ توصية الإجراء: {action}\n\n"
        
        message += f"🔍 تحليل مفصل (بناءً على وايكوف، إليوت، VSA، إيتشيموكو):\n"
        message += f"• RSI: {indicators['rsi']} (زخم { 'إيجابي' if indicators['rsi'] > 50 else 'سلبي'})\n"
        message += f"• نسبة الحجم: {indicators['volume_ratio']}x (نشاط { 'مرتفع' if indicators['volume_ratio'] > 1 else 'منخفض'})\n"
        message += f"• التقلب: {indicators['volatility']*100}% (ATR: {indicators['atr_ratio']*100}%)\n"
        message += f"• MACD Histogram: {indicators['macd_hist']} (زخم { 'إيجابي' if indicators['macd_hist'] > 0 else 'سلبي'})\n"
        message += f"• موقع Bollinger: {indicators['bb_position']*100}% (فوق/تحت الوسط)\n"
        message += f"• نسبة انتشار الحجم (VSA): {indicators['spread_volume_ratio']} (يشير إلى { 'قوة' if indicators['spread_volume_ratio'] > indicators.get('spread_volume_mean', 0) else 'ضعف'})\n"
        message += f"• اتجاه إيتشيموكو: {indicators['ichimoku_trend']} (سحابة { 'داعمة' if indicators['ichimoku_trend'] == 'صاعد' else 'مقاومة'})\n"
        message += f"• موجات إليوت: {indicators['elliott_wave']}\n"
        message += f"• الاتجاه العام: {indicators['trend']}\n\n"
        
        message += f"🕒 التوقيت: {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚠️ هذا تحليل احترافي لدعم القرار - ليس نصيحة استثمارية. قم ببحثك الخاص."
        
        chart_base64 = self._generate_price_chart(prices, coin)
        
        for attempt in range(3):  # إعادة المحاولة حتى 3 مرات
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                self.last_notification_time[coin_key] = current_time
                logger.info(f"تم إرسال إشعار احترافي لـ {coin}", extra={"coin": coin, "source": "telegram"})
                return True
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        logger.error(f"فشل إرسال إشعار لـ {coin} بعد 3 محاولات", extra={"coin": coin, "source": "telegram"})
        return False

    def _generate_price_chart(self, prices: List[float], coin: str) -> str:
        plt.figure(figsize=(8, 4))
        plt.plot(prices, label=f"{coin.upper()} Price", color='blue')
        plt.title(f"{coin.upper()} Price Trend (Last 100 Points) - تحليل احترافي")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def send_simple_analysis(self, coin: str, price: float, phase: str, signal: str):
        message = f"💰 **{coin.upper()} تحديث سريع**\n"
        message += f"💵 السعر: ${price:,.2f}\n"
        message += f"📊 المرحلة: {phase}\n"
        message += f"🎯 الإشارة: {signal}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')}"
        
        return await self._send_message(message)

    async def _send_message(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            logger.error("تكوين تلغرام غير مكتمل", extra={"coin": "N/A", "source": "telegram"})
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
                return True
            logger.error(f"فشل إرسال الرسالة: {response.status_code}", extra={"coin": "N/A", "source": "telegram"})
            return False
                
        except Exception as e:
            logger.error(f"خطأ في إرسال الرسالة: {e}", extra={"coin": "N/A", "source": "telegram"})
            return False

    async def _send_photo_with_caption(self, caption: str, photo_base64: str) -> bool:
        if not self.token or not self.chat_id:
            logger.error("تكوين تلغرام غير مكتمل", extra={"coin": "N/A", "source": "telegram"})
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
                response = await client.post(
                    f"{self.base_url}/sendPhoto",
                    data=payload,
                    files=files,
                    timeout=15.0
                )
                
            if response.status_code == 200:
                return True
            logger.error(f"فشل إرسال الصورة: {response.status_code}", extra={"coin": "N/A", "source": "telegram"})
            return False
                
        except Exception as e:
            logger.error(f"خطأ في إرسال الصورة: {e}", extra={"coin": "N/A", "source": "telegram"})
            return False

class CryptoDataFetcher:
    """جلب بيانات العملات من مصادر متعددة مع إدارة معدل الطلبات المحسنة"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.phase_analyzer = MarketPhaseAnalyzer()
        self.cache = {}
        self.rate_limit_remaining = {'coingecko': 50, 'binance': 1200}  # افتراضي
        self.rate_limit_reset = {'coingecko': 0, 'binance': 0}

    async def get_coin_data(self, coin_data: Dict[str, str]) -> Dict[str, Any]:
        cache_key = f"{coin_data['coingecko_id']}_data"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < CACHE_TTL:
            logger.info(f"جلب البيانات من التخزين المؤقت لـ {coin_data['symbol']}", 
                        extra={"coin": coin_data['symbol'], "source": "cache"})
            return self.cache[cache_key]['data']
        
        try:
            # جعل Binance المصدر الأول
            data = await self._fetch_from_binance(coin_data['binance_symbol'])
            if not data.get('prices'):
                logger.info(f"التبديل إلى CoinGecko لـ {coin_data['symbol']} بسبب فشل Binance",
                            extra={"coin": coin_data['symbol'], "source": "binance"})
                data = await self._fetch_from_coingecko(coin_data['coingecko_id'])
            
            if not data.get('prices'):
                raise ValueError("لا بيانات متاحة من أي مصدر")
            
            # تعطيل تحليل المشاعر واستخدام قيمة ثابتة
            sentiment_score = 0.5
            
            phase_analysis = self.phase_analyzer.analyze_market_phase(
                data['prices'], data['highs'], data['lows'], data['volumes'], sentiment_score
            )
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'phase_analysis': phase_analysis,
                'prices': data['prices'],
                'timestamp': current_time,
                'source': data['source']
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            logger.info(f"تم جلب البيانات لـ {coin_data['symbol']} من {data['source']}",
                        extra={"coin": coin_data['symbol'], "source": data['source']})
            return result
                
        except Exception as e:
            await notifier.send_simple_analysis(
                coin_data['symbol'],
                0,
                "غير محدد",
                f"فشل جلب البيانات: {str(e)}. جرب لاحقاً أو تحقق من الاتصال."
            )
            logger.error(f"فشل جلب البيانات لـ {coin_data['symbol']}: {e}",
                         extra={"coin": coin_data['symbol'], "source": "N/A"})
            return {
                'price': 0,
                'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار"},
                'prices': [],
                'timestamp': current_time,
                'source': 'fallback'
            }

    async def _fetch_from_coingecko(self, coin_id: str) -> Dict[str, Any]:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._update_rate_limits(response.headers, 'coingecko')
                    return {
                        'prices': [item[1] for item in data.get('prices', [])],
                        'highs': [item[1] for item in data.get('prices', [])],  # تقريبي
                        'lows': [item[1] for item in data.get('prices', [])],
                        'volumes': [item[1] for item in data.get('total_volumes', [])],
                        'source': 'coingecko'
                    }
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                    self._update_rate_limits(response.headers, 'coingecko')
                    logger.warning(f"حد معدل CoinGecko لـ {coin_id}: محاولة {attempt + 1}, الانتظار {retry_after} ثانية",
                                   extra={"coin": coin_id, "source": "coingecko"})
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"فشل CoinGecko لـ {coin_id}: {response.status_code} - {response.text}",
                                  extra={"coin": coin_id, "source": "coingecko"})
                    break
            except Exception as e:
                logger.error(f"خطأ في CoinGecko لـ {coin_id}: {e}", extra={"coin": coin_id, "source": "coingecko"})
                break
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'coingecko_failed'}

    async def _fetch_from_binance(self, symbol: str) -> Dict[str, Any]:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=100"
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._update_rate_limits(response.headers, 'binance')
                    return {
                        'prices': [float(item[4]) for item in data],
                        'highs': [float(item[2]) for item in data],
                        'lows': [float(item[3]) for item in data],
                        'volumes': [float(item[5]) for item in data],
                        'source': 'binance'
                    }
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
                    self._update_rate_limits(response.headers, 'binance')
                    logger.warning(f"حد معدل Binance لـ {symbol}: محاولة {attempt + 1}, الانتظار {retry_after} ثانية",
                                   extra={"coin": symbol, "source": "binance"})
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"فشل Binance لـ {symbol}: {response.status_code} - {response.text}",
                                  extra={"coin": symbol, "source": "binance"})
                    break
            except Exception as e:
                logger.error(f"خطأ في Binance لـ {symbol}: {e}", extra={"coin": symbol, "source": "binance"})
                break
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    async def _get_sentiment(self, coin_symbol: str) -> float:
        """دالة المشاعر معطلة بشكل كامل، تعيد قيمة ثابتة"""
        logger.info(f"تحليل المشاعر معطل بشكل كامل لـ {coin_symbol}, يتم استخدام قيمة افتراضية 0.5")
        return 0.5  # قيمة ثابتة دون محاولة الاتصال بـ Twitter

    def _update_rate_limits(self, headers, source: str):
        if source == 'coingecko':
            remaining = headers.get('x-ratelimit-remaining', self.rate_limit_remaining['coingecko'])
            reset = headers.get('x-ratelimit-reset', self.rate_limit_reset['coingecko'])
            self.rate_limit_remaining['coingecko'] = int(remaining) if remaining else 0
            self.rate_limit_reset['coingecko'] = int(reset) if reset else time.time() + 60
        elif source == 'binance':
            remaining = headers.get('x-mbx-used-weight-1m', self.rate_limit_remaining['binance'])
            self.rate_limit_remaining['binance'] = max(0, 1200 - int(remaining)) if remaining else 0
            self.rate_limit_reset['binance'] = time.time() + 60

    async def close(self):
        await self.client.aclose()

# تهيئة المكونات
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# مهمة المراقبة التلقائية
async def market_monitoring_task():
    logger.info("بدء مهمة مراقبة مراحل السوق...", extra={"coin": "N/A", "source": "system"})
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    data = await data_fetcher.get_coin_data(coin_data)
                    phase_analysis = data['phase_analysis']
                    current_price = data['price']
                    prices = data['prices']
                    
                    # ⭐ استخدام عتبة الثقة من متغير البيئة
                    if phase_analysis['confidence'] > CONFIDENCE_THRESHOLD:
                        await notifier.send_phase_alert(coin_key, phase_analysis, current_price, prices)
                    
                    logger.info(
                        f"{coin_key.upper()}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']})",
                        extra={"coin": coin_key, "source": data['source']}
                    )
                    
                    # تغيير الانتظار بين العملات إلى 10 دقائق (600 ثانية)
                    await asyncio.sleep(20)
                    
                except Exception as e:
                    logger.error(f"خطأ في تحليل {coin_key}: {e}", extra={"coin": coin_key, "source": "N/A"})
                    continue
            
            # تغيير الانتظار بين دورات المراقبة إلى 30 دقيقة (1800 ثانية)
            await asyncio.sleep(600)
            
        except Exception as e:
            logger.error(f"خطأ في مهمة المراقبة: {e}", extra={"coin": "N/A", "source": "system"})
            await asyncio.sleep(60)

# Endpoints
@app.head("/")
@app.get("/")
async def root():
    return {
        "message": "بوت مراقبة مراحل السوق - إصدار محسن مع نظريات متقدمة",
        "status": "نشط",
        "version": "8.1.0",
        "features": [
            "تحليل مراحل السوق (وايكوف + إليوت + VSA + إيتشيموكو)",
            "مصادر متعددة: Binance كمصدر أول، CoinGecko كاحتياطي",
            "إشعارات احترافية قوية لدعم القرار",
            "عملات إضافية: ADA, XRP, DOT",
            "إدارة معدل الطلبات وتسجيل محسن"
        ],
        "confidence_threshold": CONFIDENCE_THRESHOLD  # ⭐ إظهار العتبة في الاستجابة
    }

@app.get("/phase/{coin}")
async def get_coin_phase(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail="العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    
    return {
        "coin": coin,
        "price": data['price'],
        "phase_analysis": data['phase_analysis'],
        "timestamp": datetime.now().isoformat(),
        "source": data['source'],
        "confidence_threshold": CONFIDENCE_THRESHOLD  # ⭐ إظهار العتبة في الاستجابة
    }

@app.get("/alert/{coin}")
async def send_phase_alert(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail="العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    
    success = await notifier.send_phase_alert(coin, data['phase_analysis'], data['price'], data['prices'])
    
    return {
        "message": "تم إرسال الإشعار",
        "success": success,
        "phase": data['phase_analysis']['phase'],
        "confidence_threshold": CONFIDENCE_THRESHOLD  # ⭐ إظهار العتبة في الاستجابة
    }

@app.get("/status")
async def status():
    return {
        "status": "نشط",
        "monitoring": "نشط",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "rate_limits": data_fetcher.rate_limit_remaining,
        "confidence_threshold": CONFIDENCE_THRESHOLD  # ⭐ إظهار العتبة في الاستجابة
    }

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(market_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
