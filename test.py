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

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_backtest_report(self, coin: str, report: str) -> bool:
        if not self.token or not self.chat_id:
            logger.error("رمز Telegram أو معرف المحادثة غير محدد", extra={"coin": coin, "source": "telegram"})
            return False
        
        try:
            if len(report) > 4096:
                report = report[:4090] + "..."
                
            payload = {
                'chat_id': self.chat_id,
                'text': report,
                'parse_mode': 'HTML'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", json=payload, timeout=15.0)
                if response.status_code == 200:
                    logger.info(f"تم إرسال تقرير الاختبار الرجعي لـ {coin}", extra={"coin": coin, "source": "telegram"})
                    return True
                else:
                    logger.error(f"فشل إرسال تقرير الاختبار الرجعي لـ {coin}: {response.text}", extra={"coin": coin, "source": "telegram"})
                    return False
        except Exception as e:
            logger.error(f"خطأ أثناء إرسال تقرير الاختبار الرجعي لـ {coin}: {str(e)}", extra={"coin": coin, "source": "telegram"})
            return False

class CryptoDataFetcher:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
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
            
            result = {
                'price': data['prices'][-1] if data['prices'] else 0,
                'prices': data['prices'],
                'highs': data['highs'],
                'lows': data['lows'],
                'volumes': data['volumes'],
                'timestamp': current_time,
                'source': data['source']
            }
            
            self.cache[cache_key] = {'data': result, 'timestamp': current_time}
            return result
                
        except Exception as e:
            logger.error(f"فشل جلب البيانات لـ {coin_data['symbol']}: {str(e)}", extra={"coin": coin_data['symbol'], "source": "fetcher"})
            return {'price': 0, 'prices': [], 'highs': [], 'lows': [], 'volumes': [], 'timestamp': current_time, 'source': 'fallback'}

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

    async def close(self):
        await self.client.aclose()

# دالة الاختبار الرجعي
def backtest_signals(prices: List[float], highs: List[float], lows: List[float], volumes: List[float], coin_data: Dict) -> Dict[str, Any]:
    if len(prices) < 50:
        logger.error("بيانات غير كافية للاختبار الرجعي", extra={"coin": coin_data['symbol'], "source": "backtest"})
        return {
            'win_rate': 0,
            'total_return': 0,
            'number_of_trades': 0,
            'signals': []
        }
    
    try:
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
    
    except Exception as e:
        logger.error(f"خطأ في الاختبار الرجعي: {str(e)}", extra={"coin": coin_data['symbol'], "source": "backtest"})
        return {
            'win_rate': 0,
            'total_return': 0,
            'number_of_trades': 0,
            'signals': []
        }

# تهيئة
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

@app.get("/")
async def root():
    return {"message": "بوت مبسط لتحليل مراحل السوق", "version": "8.2.0"}

@app.get("/backtest/{coin}")
async def run_backtest(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(400, "غير مدعومة")
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data)
    
    if not data['prices']:
        logger.error(f"لا بيانات متاحة لـ {coin} للاختبار الرجعي", extra={"coin": coin, "source": "backtest"})
        report = (
            f"📊 **تقرير الاختبار الرجعي لـ {coin.upper()}**\n"
            f"⚠️ فشل: لا بيانات متاحة\n"
            f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}"
        )
        success = await notifier.send_backtest_report(coin, report)
        return {
            "coin": coin,
            "win_rate": 0,
            "total_return": 0,
            "number_of_trades": 0,
            "telegram_sent": success
        }
    
    result = backtest_signals(
        prices=data['prices'],
        highs=data['highs'],
        lows=data['lows'],
        volumes=data['volumes'],
        coin_data=coin_data
    )
    
    # إعداد تقرير الاختبار الرجعي
    report = (
        f"📊 **تقرير الاختبار الرجعي لـ {coin.upper()}**\n"
        f"📈 معدل الفوز: {result['win_rate']*100}%\n"
        f"💰 العائد الكلي: {result['total_return']*100}%\n"
        f"🔄 عدد الصفقات: {result['number_of_trades']}\n"
        f"🕒 {datetime.now().strftime('%H:%M %d-%m-%Y')}\n"
        f"⚠️ ليس نصيحة استثمارية."
    )
    
    # إرسال التقرير إلى Telegram
    success = await notifier.send_backtest_report(coin, report)
    
    return {
        "coin": coin,
        "win_rate": result['win_rate'],
        "total_return": result['total_return'],
        "number_of_trades": result['number_of_trades'],
        "telegram_sent": success
    }

@app.get("/status")
async def status():
    return {"status": "نشط", "supported_coins": list(SUPPORTED_COINS.keys())}

@app.on_event("shutdown")
async def shutdown_event():
    await data_fetcher.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
