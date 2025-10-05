from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
from datetime import datetime
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from logging.handlers import RotatingFileHandler

# إعداد التسجيل
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s - Coin: %(coin)s - Source: %(source)s'))
logger.addHandler(file_handler)

app = FastAPI(title="Crypto Market Phase Bot", version="8.2.0")

# إعدادات البيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", 900))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.65))

# تعريف العملات
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC',
            'rsi_low': 55, 'rsi_high': 65},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH',
            'rsi_low': 50, 'rsi_high': 70},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB',
            'rsi_low': 50, 'rsi_high': 70},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL',
            'rsi_low': 45, 'rsi_high': 75},
    'ada': {'name': 'Cardano', 'coingecko_id': 'cardano', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA',
            'rsi_low': 50, 'rsi_high': 70},
    'xrp': {'name': 'XRP', 'coingecko_id': 'ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP',
            'rsi_low': 50, 'rsi_high': 70},
    'dot': {'name': 'Polkadot', 'coingecko_id': 'polkadot', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT',
            'rsi_low': 50, 'rsi_high': 70}
}

class MarketPhaseAnalyzer:
    @staticmethod
    def analyze_market_phase(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], sentiment_score: float, coin_custom: Dict) -> Dict[str, Any]:
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
            
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            phase_analysis = MarketPhaseAnalyzer._determine_phase(latest, prev, sentiment_score, coin_custom)
            return phase_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المرحلة: {e}", extra={"coin": "N/A", "source": "N/A"})
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار"}
    
    @staticmethod
    def _determine_phase(latest, prev, sentiment_score: float, coin_custom: Dict) -> Dict[str, Any]:
        rsi_low = coin_custom.get('rsi_low', 50)
        rsi_high = coin_custom.get('rsi_high', 70)
        
        accumulation_signs = [
            latest['rsi'] < rsi_low,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.05,
            sentiment_score < 0.5,
        ]
        
        markup_signs = [
            latest['close'] > latest['sma20'] > latest['sma50'],
            latest['rsi'] > 50,
            latest['close'] > prev['close'],
            sentiment_score > 0.6,
        ]
        
        distribution_signs = [
            latest['rsi'] > rsi_high,
            abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.1,
            sentiment_score > 0.8,
        ]
        
        markdown_signs = [
            latest['close'] < latest['sma20'] < latest['sma50'],
            latest['rsi'] < 40,
            latest['close'] < prev['close'],
            sentiment_score < 0.4,
        ]
        
        scores = {
            "تجميع": sum(accumulation_signs),
            "صعود": sum(markup_signs),
            "توزيع": sum(distribution_signs),
            "هبوط": sum(markdown_signs)
        }
        
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / len(accumulation_signs)
        
        action = MarketPhaseAnalyzer._get_action_recommendation(best_phase, confidence, latest)
        
        return {
            "phase": best_phase,
            "confidence": round(confidence, 2),
            "action": action,
            "scores": scores,
            "indicators": {
                "rsi": round(latest['rsi'], 1),
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, latest) -> str:
        actions = {
            "تجميع": "مراقبة للشراء عند الكسر.",
            "صعود": "شراء على الارتدادات.",
            "توزيع": "استعداد للبيع.",
            "هبوط": "بيع على الارتدادات."
        }
        base_action = actions.get(phase, "انتظار بسبب عدم وضوح الإشارات.")
        
        if confidence > CONFIDENCE_THRESHOLD:
            if phase == "تجميع":
                return f"استعداد للشراء - مرحلة تجميع قوية. {base_action}"
            elif phase == "صعود":
                return f"شراء - اتجاه صاعد قوي. {base_action}"
            elif phase == "توزيع":
                return f"بيع - مرحلة توزيع نشطة. {base_action}"
            elif phase == "هبوط":
                return f"بيع - اتجاه هابط قوي. {base_action}"
        
        return base_action

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification_time = {}
        self.min_notification_interval = 0
        self.confidence_threshold = CONFIDENCE_THRESHOLD

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float, prices: List[float]):
        if analysis["confidence"] < self.confidence_threshold:
            logger.info(f"تم تخطي إشعار {coin}: الثقة منخفضة", 
                        extra={"coin": coin, "source": "telegram"})
            return False
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        message = f"🎯 **{coin.upper()} - مرحلة {phase}**\n"
        message += f"💰 السعر: ${price:,.2f}\n"
        message += f"📊 الثقة: {confidence*100}%\n"
        message += f"⚡ التوصية: {action}\n\n"
        message += f"🔍 تحليل:\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• اتجاه: {indicators['trend']}\n\n"
        message += f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        message += f"⚠️ ليس نصيحة استثمارية."
        
        chart_base64 = self._generate_price_chart(prices, coin)
        
        for attempt in range(3):
            success = await self._send_photo_with_caption(message, chart_base64)
            if success:
                self.last_notification_time[f"{coin}_phase"] = time.time()
                logger.info(f"تم إرسال إشعار لـ {coin}", extra={"coin": coin, "source": "telegram"})
                return True
            await asyncio.sleep(2 ** attempt)
        logger.error(f"فشل إرسال لـ {coin}", extra={"coin": coin, "source": "telegram"})
        return False

    def _generate_price_chart(self, prices: List[float], coin: str) -> str:
        plt.figure(figsize=(8, 4))
        plt.plot(prices, label=f"{coin.upper()} Price", color='blue')
        plt.title(f"{coin.upper()} Price Trend")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        return base64.b64encode(buffer.read()).decode('utf-8')

    async def _send_photo_with_caption(self, caption: str, photo_base64: str) -> bool:
        if not self.token or not self.chat_id:
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
        except Exception:
            return False

    async def send_simple_analysis(self, coin: str, price: float, phase: str, signal: str):
        if price == 0:
            logger.warning(f"تم تخطي إشعار بسيط لـ {coin} بسبب سعر 0", extra={"coin": coin, "source": "telegram"})
            return False
        
        message = f"💰 **{coin.upper()} تحديث**\n"
        message += f"💵 السعر: ${price:,.2f}\n"
        message += f"📊 المرحلة: {phase}\n"
        message += f"🎯 الإشارة: {signal}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')}"
        
        async with httpx.AsyncClient() as client:
            payload = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
            response = await client.post(f"{self.base_url}/sendMessage", json=payload, timeout=15.0)
            return response.status_code == 200

class CryptoDataFetcher:
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
            
            sentiment_score = await self._get_sentiment(coin_data['symbol'])
            
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
                    highs = [p * 1.01 for p in prices]
                    lows = [p * 0.99 for p in prices]
                    return {'prices': prices, 'highs': highs, 'lows': lows, 'volumes': volumes, 'source': 'coingecko'}
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
            except:
                pass
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
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
            except:
                pass
        return {'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'source': 'binance_failed'}

    async def _get_sentiment(self, coin_symbol: str) -> float:
        return 0.5  # قيمة افتراضية بسيطة

    async def close(self):
        await self.client.aclose()

# دالة الاختبار الرجعي
def backtest_signals(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], coin_data: Dict) -> Dict[str, Any]:
    df = pd.DataFrame({'close': prices, 'high': highs, 'low': lows, 'volume': volumes})
    
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    signals = []
    for i in range(50, len(df)):
        latest = df.iloc[i]
        prev = df.iloc[i-10] if i > 10 else df.iloc[0]
        
        buy_signal = False
        sell_signal = False
        
        if latest['sma20'] > latest['sma50']:
            buy_signal = True
        elif latest['sma20'] < latest['sma50']:
            sell_signal = True
        
        rsi_low = coin_data.get('rsi_low', 50)
        rsi_high = coin_data.get('rsi_high', 70)
        if latest['rsi'] < rsi_low:
            buy_signal = buy_signal or True
        elif latest['rsi'] > rsi_high:
            sell_signal = sell_signal or True
        
        signal = 'buy' if buy_signal else 'sell' if sell_signal else 'hold'
        signals.append({'index': i, 'signal': signal, 'price': latest['close']})
    
    trades = []
    position = None
    entry_price = 0
    for signal in signals:
        if signal['signal'] == 'buy' and position != 'long':
            position = 'long'
            entry_price = signal['price']
        elif signal['signal'] == 'sell' and position == 'long':
            trades.append(signal['price'] - entry_price)
            position = None
    
    win_rate = len([t for t in trades if t > 0]) / len(trades) if trades else 0
    total_return = sum(trades) / prices[0] if prices else 0
    
    return {
        'win_rate': round(win_rate, 2),
        'total_return': round(total_return, 4),
        'number_of_trades': len(trades),
        'signals': signals
    }

# تهيئة
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

async def market_monitoring_task():
    while True:
        for coin_key, coin_data in SUPPORTED_COINS.items():
            data = await data_fetcher.get_coin_data(coin_data)
            phase_analysis = data['phase_analysis']
            if phase_analysis['confidence'] > CONFIDENCE_THRESHOLD:
                await notifier.send_phase_alert(coin_key, phase_analysis, data['price'], data['prices'])
            await asyncio.sleep(20)
        await asyncio.sleep(600)

@app.get("/")
async def root():
    return {"message": "بوت مبسط لتحليل مراحل السوق", "version": "8.2.0"}

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
    return {"status": "نشط", "supported_coins": list(SUPPORTED_COINS.keys()), "confidence_threshold": CONFIDENCE_THRESHOLD}

@app.get("/backtest/{coin}")
async def run_backtest(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    result = backtest_signals(
        prices=data['prices'],
        highs=data['highs'],
        lows=data['lows'],
        volumes=data['volumes'],
        coin_data=coin_data
    )
    return {
        "coin": coin,
        "win_rate": result['win_rate'],
        "total_return": result['total_return'],
        "number_of_trades": result['number_of_trades']
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
