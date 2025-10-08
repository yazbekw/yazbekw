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

app = FastAPI(title="Crypto Market Phase Bot", version="9.2.0")

# ⭐ عتبة ثقة واقعية
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))
CONFIDENCE_THRESHOLD = 0.65  # 65% عتبة واقعية

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
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'dot',
            'volatility_threshold': 0.05, 'rsi_low': 50, 'rsi_high': 70}
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

class MarketPhaseAnalyzer:
    """محلل مراحل السوق مع أوزان واقعية"""
    
    # ⭐ أوزان واقعية ومتوازنة
    INDICATOR_WEIGHTS = {
        'price_trend': 1.8,      # اتجاه السعر (مهم)
        'volume_trend': 1.5,     # اتجاه الحجم
        'rsi_signal': 1.6,       # إشارة RSI
        'macd_signal': 1.4,      # إشارة MACD
        'volatility': 1.2,       # التقلبات
        'bb_signal': 1.3,        # إشارة بولينجر
        'support_resistance': 1.7, # مستويات الدعم والمقاومة
        'market_structure': 1.4,   # هيكل السوق
        'momentum': 1.3,         # الزخم
        'sentiment': 1.1         # المشاعر
    }
    
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], sentiment_score: float, coin_custom: Dict) -> Dict[str, Any]:
        if len(prices) < 50:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}}
        
        try:
            df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
            
            # المؤشرات الأساسية
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            
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
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            df['tr'] = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            # مستويات الدعم والمقاومة
            support, resistance = MarketPhaseAnalyzer._calculate_support_resistance(highs, lows, prices[-1])
            
            latest = df.iloc[-1]
            prev_5 = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            prev_10 = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            # تحليل المرحلة مع الأوزان الواقعية
            phase_analysis = MarketPhaseAnalyzer._realistic_weighted_analysis(
                latest, prev_5, prev_10, sentiment_score, support, resistance, coin_custom
            )
            
            return phase_analysis
            
        except Exception as e:
            safe_log_error(f"خطأ في تحليل المرحلة: {e}", "N/A", "analyzer")
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار", "indicators": {}}
    
    @staticmethod
    def _calculate_support_resistance(highs: List[float], lows: List[float], current_price: float) -> Tuple[float, float]:
        """حساب مستويات الدعم والمقاومة"""
        try:
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            resistance = max(recent_highs) if recent_highs else current_price * 1.08
            support = min(recent_lows) if recent_lows else current_price * 0.92
            
            return support, resistance
        except:
            return current_price * 0.92, current_price * 1.08
    
    @staticmethod
    def _realistic_weighted_analysis(latest, prev_5, prev_10, sentiment_score: float, support: float, resistance: float, coin_custom: Dict) -> Dict[str, Any]:
        """تحليل مرجح واقعي"""
        
        vol_thresh = coin_custom.get('volatility_threshold', 0.05)
        rsi_low = coin_custom.get('rsi_low', 60)
        rsi_high = coin_custom.get('rsi_high', 70)
        current_price = latest['close']
        price_change_5 = (current_price - prev_5['close']) / prev_5['close']
        price_change_10 = (current_price - prev_10['close']) / prev_10['close']
        
        # 🟢 إشارات الصعود الواقعية
        markup_signals = [
            (latest['close'] > latest['sma20'] > latest['sma50'], 1.8),
            (latest['volume_ratio'] > 1.1 and latest['close'] > prev_5['close'], 1.5),
            (latest['rsi'] > 45 and latest['rsi'] < 70, 1.6),
            (latest['macd'] > latest['macd_signal'], 1.4),
            (0.3 < latest['bb_position'] < 0.7, 1.3),
            (current_price > support * 1.02, 1.7),
            (price_change_5 > 0.02, 1.3),
            (price_change_10 > 0.05, 1.4),
            (latest['volatility'] < vol_thresh * 1.5, 1.2),
            (sentiment_score > 0.5, 1.1)
        ]
        
        # 🔴 إشارات الهبوط الواقعية
        markdown_signals = [
            (latest['close'] < latest['sma20'] < latest['sma50'], 1.8),
            (latest['volume_ratio'] > 1.1 and latest['close'] < prev_5['close'], 1.5),
            (latest['rsi'] < 45, 1.6),
            (latest['macd'] < latest['macd_signal'], 1.4),
            (latest['bb_position'] < 0.3, 1.3),
            (current_price < resistance * 0.98, 1.7),
            (price_change_5 < -0.02, 1.3),
            (price_change_10 < -0.05, 1.4),
            (latest['volatility'] > vol_thresh, 1.2),
            (sentiment_score < 0.4, 1.1)
        ]
        
        # 🟡 إشارات التجميع الواقعية
        accumulation_signals = [
            (latest['volatility'] < vol_thresh, 1.2),
            (latest['volume_ratio'] < 0.9, 1.1),
            (latest['rsi'] < rsi_low, 1.6),
            (abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.04, 1.3),
            (latest['macd_hist'] > -0.02, 1.2),
            (current_price <= support * 1.03, 1.7),
            (price_change_10 > -0.03, 1.1)
        ]
        
        # 🟠 إشارات التوزيع الواقعية
        distribution_signals = [
            (latest['volatility'] > vol_thresh, 1.2),
            (latest['volume_ratio'] > 1.3, 1.3),
            (latest['rsi'] > rsi_high, 1.6),
            (abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.06, 1.3),
            (latest['macd_hist'] < 0, 1.2),
            (current_price >= resistance * 0.97, 1.7),
            (price_change_10 > 0.08, 1.1)
        ]
        
        # حساب النقاط المرجحة
        markup_score = sum(weight for signal, weight in markup_signals if signal)
        markdown_score = sum(weight for signal, weight in markdown_signals if signal)
        accumulation_score = sum(weight for signal, weight in accumulation_signals if signal)
        distribution_score = sum(weight for signal, weight in distribution_signals if signal)
        
        scores = {
            "صعود": markup_score,
            "هبوط": markdown_score,
            "تجميع": accumulation_score,
            "توزيع": distribution_score
        }
        
        best_phase = max(scores, key=scores.get)
        best_score = scores[best_phase]
        
        # ⭐ حساب ثقة واقعية (لا تتجاوز 85%)
        total_possible_score = 16.0  # مجموع الأوزان القصوى الواقعي
        base_confidence = min(best_score / total_possible_score, 0.85)  # حد أقصى 85%
        
        # ⭐ تعديل الثقة بناءً على قوة الإشارات
        if best_score > total_possible_score * 0.7:  # إذا تجاوز 70%
            confidence = min(base_confidence * 1.1, 0.85)  # زيادة طفيفة
        elif best_score < total_possible_score * 0.4:  # إذا كان أقل من 40%
            confidence = base_confidence * 0.8  # تخفيض
        else:
            confidence = base_confidence
        
        action = MarketPhaseAnalyzer._get_action_recommendation(best_phase, confidence, current_price, support, resistance)
        
        return {
            "phase": best_phase,
            "confidence": round(confidence, 2),
            "action": action,
            "scores": scores,
            "indicators": {
                "rsi": round(latest['rsi'], 1),
                "volume_ratio": round(latest['volume_ratio'], 2),
                "volatility": round(latest['volatility'] * 100, 1),
                "macd_hist": round(latest['macd_hist'], 4),
                "bb_position": round(latest['bb_position'], 2),
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط",
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "sentiment": "إيجابي" if sentiment_score > 0.6 else "سلبي" if sentiment_score < 0.4 else "محايد",
                "price_change_5": f"{price_change_5*100:+.1f}%",
                "price_change_10": f"{price_change_10*100:+.1f}%"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, current_price: float, support: float, resistance: float) -> str:
        """توصيات واقعية"""
        
        if confidence > 0.75:
            if phase == "صعود":
                profit_potential = ((resistance - current_price) / current_price * 100)
                return f"🟢 شراء قوي - الهدف: ${resistance:,.2f} (+{profit_potential:.1f}%)"
            elif phase == "هبوط":
                risk_potential = ((support - current_price) / current_price * 100)
                return f"🔴 بيع قوي - الدعم: ${support:,.2f} ({risk_potential:+.1f}%)"
            elif phase == "تجميع":
                return f"🟡 شراء تراكمي - الدعم: ${support:,.2f}"
            elif phase == "توزيع":
                return f"🟠 بيع تدريجي - المقاومة: ${resistance:,.2f}"
        
        elif confidence > 0.6:
            if phase == "صعود":
                return f"🟢 شراء - الهدف: ${resistance:,.2f}"
            elif phase == "هبوط":
                return f"🔴 بيع - الدعم: ${support:,.2f}"
            elif phase == "تجميع":
                return f"🟡 مراقبة للشراء عند ${support:,.2f}"
            elif phase == "توزيع":
                return f"🟠 مراقبة للبيع عند ${resistance:,.2f}"
        
        else:
            return "⚪ انتظار - إشارات غير واضحة"

class TelegramNotifier:
    """إشعارات تلغرام"""
    
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
        
        safe_log_info(f"✅ تم قبول إشعار {coin}: الثقة كافية", coin, "filter")
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        message = f"🎯 **{coin.upper()} - {phase.upper()}**\n\n"
        message += f"💰 **السعر:** ${price:,.2f}\n"
        message += f"📊 **الثقة:** {confidence*100}%\n"
        message += f"⚡ **التوصية:** {action}\n\n"
        
        message += f"🔍 **التحليل:**\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• الحجم: {indicators['volume_ratio']}x\n"
        message += f"• التقلب: {indicators['volatility']}%\n"
        message += f"• MACD: {indicators['macd_hist']}\n"
        message += f"• بولينجر: {indicators['bb_position']}\n"
        message += f"• الدعم: ${indicators['support']:,.2f}\n"
        message += f"• المقاومة: ${indicators['resistance']:,.2f}\n"
        message += f"• تغير 5 أيام: {indicators['price_change_5']}\n\n"
        
        message += f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚡ v9.2 - مرشح: {self.confidence_threshold*100}%"

        chart_base64 = self._generate_price_chart(prices, coin, indicators['support'], indicators['resistance'])
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                safe_log_info(f"تم إرسال إشعار لـ {coin} بنجاح", coin, "telegram")
                return True
            await asyncio.sleep(2 ** attempt)
        
        safe_log_error(f"فشل إرسال لـ {coin}", coin, "telegram")
        return False

    def _generate_price_chart(self, prices: List[float], coin: str, support: float, resistance: float) -> str:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(prices, label='السعر', color='blue', linewidth=2)
            plt.axhline(y=support, color='green', linestyle='--', alpha=0.7, label=f'دعم: ${support:,.0f}')
            plt.axhline(y=resistance, color='red', linestyle='--', alpha=0.7, label=f'مقاومة: ${resistance:,.0f}')
            plt.title(f"{coin.upper()} - تحليل السوق v9.2")
            plt.xlabel("الفترة")
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

# باقي الكود (CryptoDataFetcher والوظائف) يبقى كما هو مع تحديث الإصدار إلى 9.2.0

class CryptoDataFetcher:
    """جلب بيانات العملات"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.phase_analyzer = MarketPhaseAnalyzer()
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
            return {'price': 0, 'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار", "indicators": {}}, 'prices': [], 'timestamp': current_time, 'source': 'fallback'}

    async def _fetch_from_coingecko(self, coin_id: str) -> Dict[str, Any]:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=100&interval=daily"
        for attempt in range(3):
            try:
                response = await self.client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    prices = [item[1] for item in data.get('prices', [])][-100:]
                    volumes = [item[1] for item in data.get('total_volumes', [])][-100:]
                    highs = [p * 1.01 for p in prices]
                    lows = [p * 0.99 for p in prices]
                    return {'prices': prices, 'highs': highs, 'lows': lows, 'volumes': volumes, 'source': 'coingecko'}
                await asyncio.sleep(2 ** attempt)
            except Exception:
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

    async def _get_sentiment_from_market(self, prices: List[float]) -> float:
        try:
            if len(prices) < 10:
                return 0.5
            
            recent_prices = prices[-10:]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            sentiment = 0.5 + (price_change * 2)
            return max(0.0, min(1.0, sentiment))
            
        except Exception:
            return 0.5

    async def close(self):
        await self.client.aclose()

# تهيئة
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def market_monitoring_task():
    """مهمة المراقبة الرئيسية"""
    safe_log_info(f"بدء مراقبة السوق - عتبة الثقة: {CONFIDENCE_THRESHOLD*100}%", "all", "monitoring")
    
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
                            safe_log_info(f"✅ تم إرسال إشعار {coin_key}", coin_key, "monitoring")
                    else:
                        safe_log_info(f"⏭️ تخطي {coin_key}: الثقة غير كافية", coin_key, "monitoring")
                    
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    safe_log_error(f"خطأ في {coin_key}: {e}", coin_key, "monitoring")
                    continue
                    
            await asyncio.sleep(300)
            
        except Exception as e:
            safe_log_error(f"خطأ في المهمة الرئيسية: {e}", "all", "monitoring")
            await asyncio.sleep(60)

@app.get("/")
async def root():
    return {"message": "بوت تحليل السوق المحسن", "version": "9.2.0", "confidence_threshold": CONFIDENCE_THRESHOLD}

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
        "status": "نشط", 
        "version": "9.2.0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "cache_size": len(data_fetcher.cache)
    }

@app.on_event("startup")
async def startup_event():
    safe_log_info(f"بدء التشغيل - v9.2.0 - عتبة الثقة: {CONFIDENCE_THRESHOLD*100}%", "system", "startup")
    asyncio.create_task(market_monitoring_task())

@app.on_event("shutdown")
async def shutdown_event():
    safe_log_info("إيقاف البوت", "system", "shutdown")
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
