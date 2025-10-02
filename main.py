from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import math
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
import json
import random
import pandas as pd
import numpy as np

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Trading Bot",
    description="Multi-crypto analysis with advanced technical indicators",
    version="5.0.0"
)

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# تعريف العملات المدعومة
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'symbol': 'SOL'},
    'link': {'name': 'Chainlink', 'coingecko_id': 'chainlink', 'symbol': 'LINK'}
}

class AdvancedDataProcessor:
    """معالج متقدم للبيانات والمؤشرات"""
    
    @staticmethod
    def calculate_advanced_indicators(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """حساب المؤشرات التقنية المتقدمة"""
        if len(prices) < 50:
            return AdvancedDataProcessor._get_default_indicators(prices[-1] if prices else 1000)
        
        try:
            # إنشاء DataFrame للبيانات
            df = pd.DataFrame({
                'close': prices,
                'volume': volumes
            })
            
            # 1. المتوسطات المتحركة
            df['sma10'] = df['close'].rolling(10).mean()
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            
            # 2. RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 3. الزخم (Momentum)
            df['momentum'] = df['close'] / df['close'].shift(5) - 1
            
            # 4. نسبة الحجم (Volume Ratio)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # 5. MACD
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp12 - exp26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # الحصول على آخر قيمة
            latest = df.iloc[-1]
            
            # تحديد شروط الشراء والبيع حسب متطلباتك
            buy_conditions = [
                latest['sma10'] > latest['sma50'],  # ✅ الاتجاه: SMA10 > SMA50
                latest['sma10'] > latest['sma20'],  # ✅ الاتجاه: SMA10 > SMA20  
                45 <= latest['rsi'] <= 68,          # ✅ الزخم: RSI بين 45-68 للشراء
                latest['momentum'] > 0,             # ✅ الزخم: momentum إيجابي
                latest['volume_ratio'] > 0.8,       # ✅ الحجم: volume ratio > 0.8
                latest['macd'] > latest['macd_signal']  # ✅ المؤشرات: MACD إيجابي
            ]
            
            sell_conditions = [
                latest['sma10'] < latest['sma50'],  # الاتجاه عكسي للبيع
                latest['sma10'] < latest['sma20'],  # الاتجاه عكسي للبيع
                32 <= latest['rsi'] <= 55,          # ✅ الزخم: RSI بين 32-55 للبيع
                latest['momentum'] < 0,             # الزخم سلبي للبيع
                latest['volume_ratio'] > 0.8,       # ✅ الحجم: volume ratio > 0.8
                latest['macd'] < latest['macd_signal']  # MACD سلبي للبيع
            ]
            
            # حساب النقاط لكل إشارة
            buy_score = sum([
                1.5 if buy_conditions[0] else 0,  # SMA10 > SMA50
                1.0 if buy_conditions[1] else 0,  # SMA10 > SMA20
                1.2 if buy_conditions[2] else 0,  # RSI شراء
                1.0 if buy_conditions[3] else 0,  # momentum إيجابي
                0.8 if buy_conditions[4] else 0,  # volume ratio
                1.0 if buy_conditions[5] else 0,  # MACD إيجابي
            ])
            
            sell_score = sum([
                1.5 if sell_conditions[0] else 0,  # SMA10 < SMA50
                1.0 if sell_conditions[1] else 0,  # SMA10 < SMA20
                1.2 if sell_conditions[2] else 0,  # RSI بيع
                1.0 if sell_conditions[3] else 0,  # momentum سلبي
                0.8 if sell_conditions[4] else 0,  # volume ratio
                1.0 if sell_conditions[5] else 0,  # MACD سلبي
            ])
            
            # تحديد الإشارة النهائية
            direction = None
            if buy_score >= 4.0:  # عتبة الشراء
                direction = "LONG"
            elif sell_score >= 4.0:  # عتبة البيع
                direction = "SHORT"
            
            return {
                'sma10': round(latest['sma10'], 4),
                'sma20': round(latest['sma20'], 4),
                'sma50': round(latest['sma50'], 4),
                'rsi': round(latest['rsi'], 2),
                'momentum': round(latest['momentum'], 4),
                'volume_ratio': round(latest['volume_ratio'], 2),
                'macd': round(latest['macd'], 4),
                'macd_signal': round(latest['macd_signal'], 4),
                'macd_histogram': round(latest['macd_histogram'], 4),
                'current_price': round(latest['close'], 4),
                'buy_score': round(buy_score, 2),
                'sell_score': round(sell_score, 2),
                'direction': direction,
                'trend_strength': round((latest['sma10'] - latest['sma50']) / latest['sma50'] * 100, 2),
                'price_vs_sma20': round((latest['close'] - latest['sma20']) / latest['sma20'] * 100, 2),
                'conditions_met': {
                    'sma10_gt_sma50': buy_conditions[0],
                    'sma10_gt_sma20': buy_conditions[1],
                    'rsi_in_buy_zone': buy_conditions[2],
                    'positive_momentum': buy_conditions[3],
                    'good_volume': buy_conditions[4],
                    'macd_positive': buy_conditions[5]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات المتقدمة: {e}")
            return AdvancedDataProcessor._get_default_indicators(prices[-1] if prices else 1000)
    
    @staticmethod
    def _get_default_indicators(current_price: float) -> Dict[str, Any]:
        """المؤشرات الافتراضية في حالة الخطأ"""
        return {
            'sma10': current_price,
            'sma20': current_price,
            'sma50': current_price,
            'rsi': 50.0,
            'momentum': 0.0,
            'volume_ratio': 1.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'current_price': current_price,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'direction': None,
            'trend_strength': 0.0,
            'price_vs_sma20': 0.0,
            'conditions_met': {}
        }

class MultiSourceDataFetcher:
    """جلب البيانات من مصادر متعددة مع المؤشرات المتقدمة"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_count = 0
        self.last_request_time = 0
        self.cache = {}
        self.cache_ttl = 300  # 5 دقائق
        self.data_processor = AdvancedDataProcessor()
        
        # مصادر البيانات البديلة
        self.data_sources = [
            self._fetch_from_coingecko,
            self._fetch_from_binance,
            self._generate_simulated_data
        ]

    async def _rate_limit(self):
        """تطبيق حدود الطلبات"""
        current_time = time.time()
        if current_time - self.last_request_time < 2:
            await asyncio.sleep(2)
        self.last_request_time = time.time()

    async def _fetch_from_coingecko(self, coin_id: str, days: int) -> Optional[Dict[str, Any]]:
        """جلب البيانات من CoinGecko"""
        try:
            await self._rate_limit()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = await self.client.get(url, headers=headers)
            
            if response.status_code == 429:
                logger.warning(f"⚠️ تم تجاوز حد الطلبات في CoinGecko للعملة {coin_id}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            return self._process_data(data, coin_id)
            
        except Exception as e:
            logger.warning(f"⚠️ فشل جلب البيانات من CoinGecko للعملة {coin_id}: {e}")
            return None

    async def _fetch_from_binance(self, coin_id: str, days: int) -> Optional[Dict[str, Any]]:
        """جلب البيانات من Binance API"""
        try:
            await self._rate_limit()
            
            binance_symbols = {
                'bitcoin': 'BTCUSDT',
                'ethereum': 'ETHUSDT',
                'binancecoin': 'BNBUSDT',
                'solana': 'SOLUSDT',
                'chainlink': 'LINKUSDT'
            }
            
            symbol = binance_symbols.get(coin_id)
            if not symbol:
                return None
                
            # جلب السعر الحالي من Binance
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                current_data = response.json()
                current_price = float(current_data['price'])
                
                return self._generate_simulated_data_based_on_price(current_price, days, coin_id)
                
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ فشل جلب البيانات من Binance للعملة {coin_id}: {e}")
            return None

    def _generate_simulated_data(self, days: int, coin_id: str) -> Dict[str, Any]:
        """إنشاء بيانات محاكاة واقعية مع المؤشرات المتقدمة"""
        logger.info(f"🔄 استخدام بيانات محاكاة واقعية للعملة {coin_id}")
        
        base_prices = {
            'bitcoin': 60000,
            'ethereum': 3500,
            'binancecoin': 600,
            'solana': 150,
            'chainlink': 18
        }
        
        base_price = base_prices.get(coin_id, 1000)
        prices = []
        volumes = []
        
        # إنشاء بيانات تاريخية محاكاة
        for i in range(days * 24):
            change = random.uniform(-0.02, 0.02)
            price = base_price * (1 + change)
            prices.append(price)
            
            volume_multipliers = {
                'bitcoin': 1.0,
                'ethereum': 0.8,
                'binancecoin': 0.3,
                'solana': 0.2,
                'chainlink': 0.1
            }
            multiplier = volume_multipliers.get(coin_id, 0.5)
            volume = random.uniform(10000000, 50000000) * multiplier
            volumes.append(volume)
            
            base_price = price
        
        coin_info = self._get_coin_info(coin_id)
        
        # حساب المؤشرات المتقدمة
        indicators = self.data_processor.calculate_advanced_indicators(prices, volumes)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'current_price': prices[-1] if prices else base_price,
            'current_volume': volumes[-1] if volumes else 25000000,
            'source': 'simulated',
            'timestamp': datetime.now().isoformat(),
            'coin_id': coin_id,
            'coin_name': coin_info['name'],
            'coin_symbol': coin_info['symbol'],
            'advanced_indicators': indicators
        }

    def _generate_simulated_data_based_on_price(self, current_price: float, days: int, coin_id: str) -> Dict[str, Any]:
        """إنشاء بيانات محاكاة بناءً على سعر حقيقي"""
        prices = []
        volumes = []
        
        start_price = current_price * random.uniform(0.8, 0.95)
        
        for i in range(days * 24):
            progress = i / (days * 24)
            target_price = start_price + (current_price - start_price) * progress
            volatility = 0.01 * (1 - progress)
            price = target_price * (1 + random.uniform(-volatility, volatility))
            prices.append(price)
            
            volume_multipliers = {
                'bitcoin': 1.0,
                'ethereum': 0.8,
                'binancecoin': 0.3,
                'solana': 0.2,
                'chainlink': 0.1
            }
            multiplier = volume_multipliers.get(coin_id, 0.5)
            volume = random.uniform(10000000, 50000000) * multiplier
            volumes.append(volume)
        
        coin_info = self._get_coin_info(coin_id)
        
        # حساب المؤشرات المتقدمة
        indicators = self.data_processor.calculate_advanced_indicators(prices, volumes)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'current_price': current_price,
            'current_volume': volumes[-1] if volumes else 25000000,
            'source': 'binance_simulated',
            'timestamp': datetime.now().isoformat(),
            'coin_id': coin_id,
            'coin_name': coin_info['name'],
            'coin_symbol': coin_info['symbol'],
            'advanced_indicators': indicators
        }

    def _get_coin_info(self, coin_id: str) -> Dict[str, str]:
        """الحصول على معلومات العملة"""
        for coin_key, coin_data in SUPPORTED_COINS.items():
            if coin_data['coingecko_id'] == coin_id:
                return coin_data
        return {'name': coin_id, 'symbol': coin_id.upper()}

    def _process_data(self, data: Dict[str, Any], coin_id: str) -> Dict[str, Any]:
        """معالجة البيانات من CoinGecko"""
        coin_info = self._get_coin_info(coin_id)
        
        prices = [item[1] for item in data['prices']]
        volumes = [item[1] for item in data['total_volumes']]
        
        # حساب المؤشرات المتقدمة
        indicators = self.data_processor.calculate_advanced_indicators(prices, volumes)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'current_price': data['prices'][-1][1] if data['prices'] else 0,
            'current_volume': data['total_volumes'][-1][1] if data['total_volumes'] else 0,
            'source': 'coingecko',
            'timestamp': datetime.now().isoformat(),
            'coin_id': coin_id,
            'coin_name': coin_info['name'],
            'coin_symbol': coin_info['symbol'],
            'advanced_indicators': indicators
        }

    async def get_coin_data(self, coin_id: str, days: int = 30) -> Dict[str, Any]:
        """جلب البيانات من أفضل مصدر متاح"""
        cache_key = f"{coin_id}_data_{days}"
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                logger.info(f"✅ استخدام البيانات من الذاكرة المؤقتة للعملة {coin_id}")
                return cached_data
        
        for source in self.data_sources:
            try:
                if asyncio.iscoroutinefunction(source):
                    data = await source(coin_id, days)
                else:
                    data = source(days, coin_id)
                    
                if data is not None:
                    logger.info(f"✅ تم جلب البيانات من {data.get('source', 'unknown')} للعملة {coin_id}")
                    self.cache[cache_key] = (data, current_time)
                    return data
                    
            except Exception as e:
                logger.warning(f"⚠️ فشل المصدر للعملة {coin_id}: {e}")
                continue
        
        logger.warning(f"🔄 استخدام البيانات المحاكاة بعد فشل جميع المصادر للعملة {coin_id}")
        data = self._generate_simulated_data(days, coin_id)
        self.cache[cache_key] = (data, current_time)
        return data

    async def close(self):
        """إغلاق العميل"""
        await self.client.aclose()

class AdvancedCryptoAnalyzer:
    """محلل متقدم للعملات المشفرة"""
    
    def __init__(self):
        self.data_fetcher = MultiSourceDataFetcher()
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'data_source_usage': {},
            'last_successful_analysis': None,
            'coins_analyzed': list(SUPPORTED_COINS.keys())
        }

    async def analyze_coin(self, coin: str) -> Dict[str, Any]:
        """تحليل عملة محددة بالمؤشرات المتقدمة"""
        try:
            logger.info(f"🔍 بدء تحليل متقدم لـ {coin}...")
            
            if coin not in SUPPORTED_COINS:
                raise HTTPException(status_code=400, detail=f"العملة {coin} غير مدعومة")
            
            coin_id = SUPPORTED_COINS[coin]['coingecko_id']
            
            # جلب البيانات
            data = await self.data_fetcher.get_coin_data(coin_id, 30)
            self.performance_stats['total_analyses'] += 1
            
            # تحديث إحصائيات مصدر البيانات
            source = data.get('source', 'unknown')
            self.performance_stats['data_source_usage'][source] = \
                self.performance_stats['data_source_usage'].get(source, 0) + 1
            
            # استخدام المؤشرات المتقدمة
            indicators = data.get('advanced_indicators', {})
            
            # تحديد الإشارة النهائية
            overall_signal = self._determine_advanced_signal(indicators)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'coin': coin,
                'coin_name': data.get('coin_name', SUPPORTED_COINS[coin]['name']),
                'coin_symbol': data.get('coin_symbol', SUPPORTED_COINS[coin]['symbol']),
                'price': round(data['current_price'], 2),
                'volume': round(data['current_volume'], 2),
                'data_source': source,
                'advanced_indicators': indicators,
                'overall_signal': overall_signal,
                'reliability': 'high' if source == 'coingecko' else 'medium',
                'analysis_id': f"ADV_{coin.upper()}_{int(time.time())}"
            }
            
            self.performance_stats['successful_analyses'] += 1
            self.performance_stats['last_successful_analysis'] = datetime.now()
            
            logger.info(f"✅ تحليل متقدم ناجح لـ {coin} - الإشارة: {overall_signal}")
            return analysis
            
        except Exception as e:
            self.performance_stats['failed_analyses'] += 1
            logger.error(f"❌ فشل في تحليل {coin}: {e}")
            return await self._get_fallback_analysis(coin)

    def _determine_advanced_signal(self, indicators: Dict[str, Any]) -> str:
        """تحديد الإشارة بناءً على المؤشرات المتقدمة"""
        direction = indicators.get('direction')
        buy_score = indicators.get('buy_score', 0)
        sell_score = indicators.get('sell_score', 0)
        
        if direction == "LONG":
            if buy_score >= 5.0:
                return "شراء قوي"
            elif buy_score >= 4.0:
                return "شراء"
            else:
                return "شراء ضعيف"
        elif direction == "SHORT":
            if sell_score >= 5.0:
                return "بيع قوي"
            elif sell_score >= 4.0:
                return "بيع"
            else:
                return "بيع ضعيف"
        else:
            return "محايد"

    async def analyze_all_coins(self) -> Dict[str, Any]:
        """تحليل جميع العملات المدعومة"""
        logger.info("🔍 بدء تحليل متقدم لجميع العملات...")
        
        analyses = {}
        tasks = []
        
        for coin in SUPPORTED_COINS.keys():
            task = asyncio.create_task(self.analyze_coin(coin))
            tasks.append((coin, task))
        
        for coin, task in tasks:
            try:
                analysis = await task
                analyses[coin] = analysis
            except Exception as e:
                logger.error(f"❌ فشل في تحليل {coin}: {e}")
                analyses[coin] = await self._get_fallback_analysis(coin)
        
        # حساب إشارة عامة
        overall_signal = self._calculate_overall_signal(analyses)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_signal': overall_signal,
            'coins_analyzed': len(analyses),
            'analyses': analyses
        }

    def _calculate_overall_signal(self, analyses: Dict[str, Any]) -> str:
        """حساب الإشارة العامة بناءً على جميع التحليلات"""
        signals = {
            "شراء قوي": 2,
            "شراء": 1,
            "شراء ضعيف": 0.5,
            "محايد": 0,
            "بيع ضعيف": -0.5,
            "بيع": -1,
            "بيع قوي": -2
        }
        
        total_score = 0
        valid_analyses = 0
        
        for coin, analysis in analyses.items():
            signal = analysis.get('overall_signal', 'محايد')
            if signal in signals:
                total_score += signals[signal]
                valid_analyses += 1
        
        if valid_analyses == 0:
            return "محايد"
        
        average_score = total_score / valid_analyses
        
        if average_score >= 1.5:
            return "شراء قوي"
        elif average_score >= 0.5:
            return "شراء"
        elif average_score <= -1.5:
            return "بيع قوي"
        elif average_score <= -0.5:
            return "بيع"
        else:
            return "محايد"

    async def _get_fallback_analysis(self, coin: str) -> Dict[str, Any]:
        """تحليل بديل عند الفشل"""
        fallback_prices = {
            'btc': 61750.0,
            'eth': 3500.0,
            'bnb': 600.0,
            'sol': 150.0,
            'link': 18.0
        }
        
        fallback_price = fallback_prices.get(coin, 100.0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'coin': coin,
            'coin_name': SUPPORTED_COINS[coin]['name'],
            'coin_symbol': SUPPORTED_COINS[coin]['symbol'],
            'price': fallback_price,
            'volume': 25000000,
            'data_source': 'fallback',
            'advanced_indicators': AdvancedDataProcessor._get_default_indicators(fallback_price),
            'overall_signal': 'محايد',
            'reliability': 'low',
            'analysis_id': f"FBA_{coin.upper()}_{int(time.time())}",
            'note': 'هذا تحليل افتراضي بسبب مشكلة تقنية'
        }

    async def close(self):
        """إغلاق الموارد"""
        await self.data_fetcher.close()

class TelegramNotifier:
    """إشعارات تلغرام مبسطة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_message(self, message: str) -> bool:
        """إرسال رسالة إلى تلغرام"""
        if not self.token or not self.chat_id:
            logger.warning("⚠️ إعدادات التلغرام غير مكتملة")
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
                logger.info("✅ تم إرسال الرسالة إلى التلغرام")
                return True
            else:
                logger.error(f"❌ خطأ في إرسال الرسالة: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التلغرام: {e}")
            return False

# تهيئة المكونات
analyzer = AdvancedCryptoAnalyzer()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# المهمة التلقائية
async def auto_analysis_task():
    """مهمة التحليل التلقائي المتقدم"""
    logger.info("🚀 بدء مهمة التحليل التلقائي المتقدم...")
    
    while True:
        try:
            # تحليل جميع العملات
            all_analyses = await analyzer.analyze_all_coins()
            
            # إنشاء رسالة التقرير المفصلة
            message = f"📊 **تقرير تحليل متقدم للعملات المشفرة**\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            message += f"🎯 الإشارة العامة: {all_analyses['overall_signal']}\n"
            message += f"💰 العملات المحللة: {all_analyses['coins_analyzed']}\n\n"
            
            message += "**التحليلات التفصيلية:**\n"
            for coin, analysis in all_analyses['analyses'].items():
                indicators = analysis['advanced_indicators']
                conditions = indicators.get('conditions_met', {})
                
                message += f"\n💰 **{analysis['coin_name']} ({analysis['coin_symbol']})**\n"
                message += f"💵 السعر: ${analysis['price']:,.2f}\n"
                message += f"📊 الإشارة: {analysis['overall_signal']}\n"
                message += f"📈 نقاط الشراء: {indicators.get('buy_score', 0)}/6.5\n"
                message += f"📉 نقاط البيع: {indicators.get('sell_score', 0)}/6.5\n\n"
                
                message += f"**المؤشرات المتقدمة:**\n"
                message += f"• RSI: {indicators.get('rsi', 'N/A')}\n"
                message += f"• SMA10: {indicators.get('sma10', 'N/A')}\n"
                message += f"• SMA20: {indicators.get('sma20', 'N/A')}\n"
                message += f"• SMA50: {indicators.get('sma50', 'N/A')}\n"
                message += f"• الزخم: {indicators.get('momentum', 'N/A')}\n"
                message += f"• الحجم: {indicators.get('volume_ratio', 'N/A')}x\n"
                message += f"• MACD: {indicators.get('macd', 'N/A')}\n\n"
                
                message += f"**الشروط المحققة:**\n"
                message += f"• SMA10 > SMA50: {'✅' if conditions.get('sma10_gt_sma50') else '❌'}\n"
                message += f"• SMA10 > SMA20: {'✅' if conditions.get('sma10_gt_sma20') else '❌'}\n"
                message += f"• RSI في النطاق: {'✅' if conditions.get('rsi_in_buy_zone') else '❌'}\n"
                message += f"• زخم إيجابي: {'✅' if conditions.get('positive_momentum') else '❌'}\n"
                message += f"• حجم جيد: {'✅' if conditions.get('good_volume') else '❌'}\n"
                message += f"• MACD إيجابي: {'✅' if conditions.get('macd_positive') else '❌'}\n"
                message += f"📡 المصدر: {analysis['data_source']}\n"
            
            message += f"\n🆔 رقم التقرير: ADV_{int(time.time())}\n"
            message += "\n⚠️ تحليل فني متقدم - ليس نصيحة استثمارية"
            
            # إرسال التقرير
            await notifier.send_message(message)
            
            logger.info(f"✅ تم إكمال دورة التحليل المتقدم - الإشارة العامة: {all_analyses['overall_signal']}")
            
            # الانتظار 30 دقيقة
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة التلقائية: {e}")
            await asyncio.sleep(60)

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "مرحباً في Advanced Crypto Trading Bot",
        "status": "نشط",
        "version": "5.0.0",
        "supported_coins": SUPPORTED_COINS,
        "features": [
            "المؤشرات المتقدمة (SMA, RSI, MACD, Momentum, Volume)",
            "نظام ترجيح متقدم للإشارات",
            "شروط دخول محسنة",
            "تحليل متعدد المصادر",
            "إشعارات تلقائية مفصلة"
        ],
        "performance": analyzer.performance_stats
    }

@app.get("/analysis/{coin}")
async def get_coin_analysis(coin: str):
    """الحصول على التحليل المتقدم لعملة محددة"""
    return await analyzer.analyze_coin(coin.lower())

@app.get("/analysis")
async def get_all_analysis():
    """الحصول على تحليل جميع العملات"""
    return await analyzer.analyze_all_coins()

@app.get("/coins")
async def get_supported_coins():
    """الحصول على قائمة العملات المدعومة"""
    return {
        "supported_coins": SUPPORTED_COINS,
        "total_coins": len(SUPPORTED_COINS)
    }

@app.get("/health")
async def health_check():
    """فحص الصحة"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "performance": analyzer.performance_stats,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "supported_coins_count": len(SUPPORTED_COINS)
    }

@app.post("/send-report")
async def send_report():
    """إرسال تقرير يدوي"""
    all_analyses = await analyzer.analyze_all_coins()
    
    message = f"📊 **تقرير يدوي متقدم للعملات المشفرة**\n"
    message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    message += f"🎯 الإشارة العامة: {all_analyses['overall_signal']}\n\n"
    
    for coin, analysis in all_analyses['analyses'].items():
        indicators = analysis['advanced_indicators']
        message += f"💰 **{analysis['coin_symbol']}**: ${analysis['price']:,.2f} - {analysis['overall_signal']}\n"
        message += f"   📊 RSI: {indicators.get('rsi', 'N/A')} | "
        message += f"📈 نقاط: {indicators.get('buy_score', 0)}/{indicators.get('sell_score', 0)} | "
        message += f"🎯 اتجاه: {indicators.get('direction', 'N/A')}\n"
    
    success = await notifier.send_message(message)
    return {"message": "تم إرسال التقرير", "success": success}

# بدء المهمة التلقائية
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_analysis_task())

@app.on_event("shutdown")
async def shutdown_event():
    await analyzer.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
