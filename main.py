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

app = FastAPI(title="Crypto Market Phase Bot", version="7.1.0")

# إعدادات التلغرام والبيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))  # 15 دقيقة لتجنب إرهاق API

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
    """محلل مراحل السوق بناءً على نظرية وايكوف مع مؤشرات إضافية"""
    
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل مرحلة السوق الحالية"""
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
            
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            phase_analysis = MarketPhaseAnalyzer._determine_phase(latest, prev, df)
            return phase_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المرحلة: {e}", extra={"coin": "N/A", "source": "N/A"})
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار"}
    
    @staticmethod
    def _determine_phase(latest, prev, df) -> Dict[str, Any]:
        """تحديد المرحلة بناءً على المؤشرات الموسعة"""
        accumulation_signs = [
            latest['volatility'] < 0.05,
            latest['volume_ratio'] < 1.2,
            latest['rsi'] < 60,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.05,
            latest['macd_hist'] > 0,
            latest['close'] > latest['bb_lower'],
            latest['atr'] / latest['close'] < 0.03  # تقلبات ATR منخفضة
        ]
        
        markup_signs = [
            latest['close'] > latest['sma20'] > latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] > 50,
            latest['close'] > prev['close'],
            latest['macd'] > latest['macd_signal'],
            latest['close'] > latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.02
        ]
        
        distribution_signs = [
            latest['volatility'] > 0.08,
            latest['volume_ratio'] > 1.5,
            latest['rsi'] > 70,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.1,
            latest['macd_hist'] < 0,
            latest['close'] < latest['bb_upper'],
            latest['atr'] / latest['close'] > 0.04
        ]
        
        markdown_signs = [
            latest['close'] < latest['sma20'] < latest['sma50'],
            latest['volume_ratio'] > 1.0,
            latest['rsi'] < 40,
            latest['close'] < prev['close'],
            latest['macd'] < latest['macd_signal'],
            latest['close'] < latest['bb_middle'],
            latest['atr'] / latest['close'] > 0.03
        ]
        
        scores = {
            "تجميع": sum(accumulation_signs),
            "صعود": sum(markup_signs),
            "توزيع": sum(distribution_signs),
            "هبوط": sum(markdown_signs)
        }
        
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / 7.0  # معدلة لـ 7 مؤشرات
        
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
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, latest) -> str:
        actions = {
            "تجميع": "مراقبة للشراء عند الكسر",
            "صعود": "شراء على الارتدادات",
            "توزيع": "استعداد للبيع",
            "هبوط": "بيع على الارتدادات"
        }
        base_action = actions.get(phase, "انتظار")
        
        if confidence > 0.75:  # زيادة الحد الأدنى للثقة
            if phase == "تجميع":
                return "استعداد للشراء - مرحلة تجميع قوية"
            elif phase == "صعود":
                return "شراء - اتجاه صاعد قوي"
            elif phase == "توزيع":
                return "بيع - مرحلة توزيع نشطة"
            elif phase == "هبوط":
                return "بيع - اتجاه هابط قوي"
        return base_action

class TelegramNotifier:
    """إشعارات تلغرام محسنة مع إعادة المحاولة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification_time = {}
        self.min_notification_interval = 10800  # 3 ساعات لتقليل الإشعارات
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        current_time = time.time()
        coin_key = f"{coin}_phase"
        
        if (coin_key in self.last_notification_time and 
            current_time - self.last_notification_time[coin_key] < self.min_notification_interval):
            logger.info(f"تم تخطي إشعار {coin}: الإشعار متكرر", extra={"coin": coin, "source": "telegram"})
            return False
        
        if analysis["confidence"] < self.confidence_threshold:
            logger.info(f"تم تخطي إشعار {coin}: الثقة منخفضة ({analysis['confidence']})", 
                        extra={"coin": coin, "source": "telegram"})
            return False
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        message = f"🎯 **{coin.upper()} - مرحلة {phase}**\n"
        message += f"💰 السعر: ${price:,.2f}\n"
        message += f"📊 الثقة: {confidence*100}%\n"
        message += f"⚡ الإجراء: {action}\n\n"
        message += f"📈 المؤشرات:\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• الحجم: {indicators['volume_ratio']}x\n"
        message += f"• التقلب: {indicators['volatility']*100}%\n"
        message += f"• MACD Hist: {indicators['macd_hist']}\n"
        message += f"• BB Position: {indicators['bb_position']*100}%\n"
        message += f"• ATR Ratio: {indicators['atr_ratio']*100}%\n"
        message += f"• الاتجاه: {indicators['trend']}\n\n"
        message += f"🕒 {datetime.now().strftime('%H:%M')}\n"
        message += "⚠️ مراقبة فقط - ليس نصيحة استثمارية"
        
        chart_base64 = self._generate_price_chart(prices, coin)
        
        for attempt in range(3):  # إعادة المحاولة حتى 3 مرات
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                self.last_notification_time[coin_key] = current_time
                logger.info(f"تم إرسال إشعار لـ {coin}", extra={"coin": coin, "source": "telegram"})
                return True
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        logger.error(f"فشل إرسال إشعار لـ {coin} بعد 3 محاولات", extra={"coin": coin, "source": "telegram"})
        return False

    def _generate_price_chart(self, prices: List[float], coin: str) -> str:
        plt.figure(figsize=(8, 4))
        plt.plot(prices, label=f"{coin.upper()} Price", color='blue')
        plt.title(f"{coin.upper()} Price Trend (Last 100 Points)")
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
    """جلب بيانات العملات من مصادر متعددة مع إدارة معدل الطلبات"""
    
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
            # التحقق من حدود معدل CoinGecko
            if self.rate_limit_remaining['coingecko'] > 5 and current_time > self.rate_limit_reset['coingecko']:
                data = await self._fetch_from_coingecko(coin_data['coingecko_id'])
            else:
                logger.warning(f"حد معدل CoinGecko منخفض، التبديل إلى Binance لـ {coin_data['symbol']}",
                              extra={"coin": coin_data['symbol'], "source": "coingecko"})
                data = await self._fetch_from_binance(coin_data['binance_symbol'])
            
            if not data.get('prices'):
                raise ValueError("لا بيانات متاحة")
            
            phase_analysis = self.phase_analyzer.analyze_market_phase(
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
            logger.info(f"تم جلب البيانات لـ {coin_data['symbol']} من {data['source']}",
                        extra={"coin": coin_data['symbol'], "source": data['source']})
            return result
                
        except Exception as e:
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
                        'highs': [item[1] for item in data.get('prices', [])],  # CoinGecko لا يوفر highs/lows، استخدام السعر كبديل
                        'lows': [item[1] for item in data.get('prices', [])],
                        'volumes': [item[1] for item in data.get('total_volumes', [])],
                        'source': 'coingecko'
                    }
                elif response.status_code == 429:
                    self._update_rate_limits(response.headers, 'coingecko')
                    logger.warning(f"حد معدل CoinGecko: محاولة {attempt + 1}", extra={"coin": coin_id, "source": "coingecko"})
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"فشل CoinGecko: {response.status_code}", extra={"coin": coin_id, "source": "coingecko"})
                    break
            except Exception as e:
                logger.error(f"خطأ في CoinGecko: {e}", extra={"coin": coin_id, "source": "coingecko"})
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
                        'prices': [float(item[4]) for item in data],  # Close price
                        'highs': [float(item[2]) for item in data],   # High price
                        'lows': [float(item[3]) for item in data],    # Low price
                        'volumes': [float(item[5]) for item in data], # Volume
                        'source': 'binance'
                    }
                elif response.status_code == 429:
                    self._update_rate_limits(response.headers, 'binance')
                    logger.warning(f"حد معدل Binance: محاولة {attempt + 1}", extra={"coin": symbol, "source": "binance"})
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"فشل Binance: {response.status_code}", extra={"coin": symbol, "source": "binance"})
                    break
            except Exception as e:
                logger.error(f"خطأ في Binance: {e}", extra={"coin": symbol, "source": "binance"})
                break
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

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
                    
                    if phase_analysis['confidence'] > 0.75:
                        await notifier.send_phase_alert(coin_key, phase_analysis, current_price, prices)
                    
                    logger.info(
                        f"{coin_key.upper()}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']})",
                        extra={"coin": coin_key, "source": data['source']}
                    )
                    
                    await asyncio.sleep(15)  # زيادة الانتظار لتجنب إرهاق API
                    
                except Exception as e:
                    logger.error(f"خطأ في تحليل {coin_key}: {e}", extra={"coin": coin_key, "source": "N/A"})
                    continue
            
            await asyncio.sleep(7200)  # 2 ساعة بين الدورات
            
        except Exception as e:
            logger.error(f"خطأ في مهمة المراقبة: {e}", extra={"coin": "N/A", "source": "system"})
            await asyncio.sleep(60)

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "بوت مراقبة مراحل السوق",
        "status": "نشط",
        "version": "7.1.0",
        "features": [
            "تحليل مراحل السوق (تجميع، صعود، توزيع، هبوط)",
            "مصادر متعددة: CoinGecko وBinance",
            "مؤشرات إضافية: MACD, Bollinger Bands, ATR",
            "إشعارات تلغرام مع رسوم بيانية وإعادة محاولة",
            "عملات إضافية: ADA, XRP, DOT",
            "إدارة معدل الطلبات وتسجيل محسن"
        ]
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
        "source": data['source']
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
        "phase": data['phase_analysis']['phase']
    }

@app.get("/status")
async def status():
    return {
        "status": "نشط",
        "monitoring": "نشط",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "rate_limits": data_fetcher.rate_limit_remaining
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
