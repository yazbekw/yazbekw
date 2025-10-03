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

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Market Phase Bot", version="6.0.0")

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# تعريف العملات
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'coingecko_id': 'bitcoin', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'coingecko_id': 'ethereum', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coingecko_id': 'binancecoin', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'coingecko_id': 'solana', 'symbol': 'SOL'}
}

class MarketPhaseAnalyzer:
    """محلل مراحل السوق بناءً على نظرية وايكوف"""
    
    @staticmethod
    def analyze_market_phase(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل مرحلة السوق الحالية"""
        if len(prices) < 20:
            return {"phase": "غير محدد", "confidence": 0, "action": "انتظار"}
        
        try:
            df = pd.DataFrame({'close': prices, 'volume': volumes})
            
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
            
            latest = df.iloc[-1]
            prev = df.iloc[-10] if len(df) > 10 else df.iloc[0]
            
            # تحديد المرحلة
            phase_analysis = MarketPhaseAnalyzer._determine_phase(latest, prev, df)
            return phase_analysis
            
        except Exception as e:
            logger.error(f"خطأ في تحليل المرحلة: {e}")
            return {"phase": "خطأ", "confidence": 0, "action": "انتظار"}
    
    @staticmethod
    def _determine_phase(latest, prev, df) -> Dict[str, Any]:
        """تحديد المرحلة بناءً على المؤشرات"""
        
        # 1. مرحلة التجميع (Accumulation)
        accumulation_signs = [
            latest['volatility'] < 0.05,  # تقلبات منخفضة
            latest['volume_ratio'] < 1.2,  # حجم معتدل
            latest['rsi'] < 60,  # RSI ليس في منطقة ذروة شراء
            abs(latest['close'] - latest['sma20']) / latest['sma20'] < 0.05  # سعر قريب من المتوسط
        ]
        
        # 2. مرحلة الصعود (Mark-Up)
        markup_signs = [
            latest['close'] > latest['sma20'] > latest['sma50'],  # اتجاه صاعد
            latest['volume_ratio'] > 1.0,  # حجم جيد
            latest['rsi'] > 50,  # زخم إيجابي
            latest['close'] > prev['close']  # سعر أعلى من السابق
        ]
        
        # 3. مرحلة التوزيع (Distribution)
        distribution_signs = [
            latest['volatility'] > 0.08,  # تقلبات عالية
            latest['volume_ratio'] > 1.5,  # حجم مرتفع
            latest['rsi'] > 70,  # RSI في منطقة ذروة شراء
            abs(latest['close'] - latest['sma20']) / latest['sma20'] > 0.1  # سعر بعيد عن المتوسط
        ]
        
        # 4. مرحلة الهبوط (Mark-Down)
        markdown_signs = [
            latest['close'] < latest['sma20'] < latest['sma50'],  # اتجاه هابط
            latest['volume_ratio'] > 1.0,  # حجم بيع جيد
            latest['rsi'] < 40,  # زخم سلبي
            latest['close'] < prev['close']  # سعر أقل من السابق
        ]
        
        # حساب النقاط لكل مرحلة
        accumulation_score = sum(accumulation_signs)
        markup_score = sum(markup_signs)
        distribution_score = sum(distribution_signs)
        markdown_score = sum(markdown_signs)
        
        scores = {
            "تجميع": accumulation_score,
            "صعود": markup_score,
            "توزيع": distribution_score,
            "هبوط": markdown_score
        }
        
        # تحديد المرحلة ذات الأعلى نقاط
        best_phase = max(scores, key=scores.get)
        confidence = scores[best_phase] / 4.0  # ثقة من 0-1
        
        # تحديد الإجراء المناسب
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
                "trend": "صاعد" if latest['sma20'] > latest['sma50'] else "هابط"
            }
        }
    
    @staticmethod
    def _get_action_recommendation(phase: str, confidence: float, latest) -> str:
        """تحديد الإجراء المناسب للمرحلة"""
        actions = {
            "تجميع": "مراقبة للشراء عند الكسر",
            "صعود": "شراء على الارتدادات",
            "توزيع": "استعداد للبيع",
            "هبوط": "بيع على الارتدادات"
        }
        
        base_action = actions.get(phase, "انتظار")
        
        if confidence > 0.7:
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
    """إشعارات تلغرام مبسطة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_notification_time = {}
        self.min_notification_interval = 3600  # ساعة واحدة بين الإشعارات لنفس العملة

    async def send_phase_alert(self, coin: str, analysis: Dict[str, Any], price: float):
        """إرسال إشعار مرحلة السوق"""
        current_time = time.time()
        coin_key = f"{coin}_phase"
        
        # التحقق من عدم إرسال إشعارات متكررة
        if (coin_key in self.last_notification_time and 
            current_time - self.last_notification_time[coin_key] < self.min_notification_interval):
            return False
        
        phase = analysis["phase"]
        confidence = analysis["confidence"]
        action = analysis["action"]
        indicators = analysis["indicators"]
        
        # إنشاء رسالة مختصرة
        message = f"🎯 **{coin.upper()} - مرحلة {phase}**\n"
        message += f"💰 السعر: ${price:,.2f}\n"
        message += f"📊 الثقة: {confidence*100}%\n"
        message += f"⚡ الإجراء: {action}\n\n"
        
        message += f"📈 المؤشرات:\n"
        message += f"• RSI: {indicators['rsi']}\n"
        message += f"• الحجم: {indicators['volume_ratio']}x\n"
        message += f"• التقلب: {indicators['volatility']*100}%\n"
        message += f"• الاتجاه: {indicators['trend']}\n\n"
        
        message += f"🕒 {datetime.now().strftime('%H:%M')}\n"
        message += "⚠️ مراقبة فقط - ليس نصيحة استثمارية"
        
        success = await self._send_message(message)
        if success:
            self.last_notification_time[coin_key] = current_time
        return success

    async def send_simple_analysis(self, coin: str, price: float, phase: str, signal: str):
        """إرسال تحليل مختصر"""
        message = f"💰 **{coin.upper()} تحديث سريع**\n"
        message += f"💵 السعر: ${price:,.2f}\n"
        message += f"📊 المرحلة: {phase}\n"
        message += f"🎯 الإشارة: {signal}\n"
        message += f"⏰ {datetime.now().strftime('%H:%M')}"
        
        return await self._send_message(message)

    async def _send_message(self, message: str) -> bool:
        """إرسال رسالة إلى تلغرام"""
        if not self.token or not self.chat_id:
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
                
            return response.status_code == 200
                
        except Exception as e:
            logger.error(f"خطأ في إرسال التلغرام: {e}")
            return False

class CryptoDataFetcher:
    """جلب بيانات العملات"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.phase_analyzer = MarketPhaseAnalyzer()
        self.cache = {}
        self.cache_ttl = 300  # 5 دقائق

    async def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        """جلب بيانات العملة"""
        cache_key = f"{coin_id}_data"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < self.cache_ttl:
            return self.cache[cache_key]['data']
        
        try:
            # جلب البيانات من CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                prices = [item[1] for item in data['prices'][-100:]]  # آخر 100 نقطة
                volumes = [item[1] for item in data['total_volumes'][-100:]]
                
                # تحليل المرحلة
                phase_analysis = self.phase_analyzer.analyze_market_phase(prices, volumes)
                
                result = {
                    'price': prices[-1] if prices else 0,
                    'phase_analysis': phase_analysis,
                    'timestamp': current_time,
                    'source': 'coingecko'
                }
                
                self.cache[cache_key] = {'data': result, 'timestamp': current_time}
                return result
                
        except Exception as e:
            logger.warning(f"فشل جلب البيانات لـ {coin_id}: {e}")
        
        # بيانات افتراضية في حالة الخطأ
        return {
            'price': 1000,
            'phase_analysis': {"phase": "غير محدد", "confidence": 0, "action": "انتظار"},
            'timestamp': current_time,
            'source': 'fallback'
        }

    async def close(self):
        await self.client.aclose()

# تهيئة المكونات
data_fetcher = CryptoDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

# مهمة المراقبة التلقائية
async def market_monitoring_task():
    """مهمة مراقبة السوق التلقائية"""
    logger.info("بدء مهمة مراقبة مراحل السوق...")
    
    while True:
        try:
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    # جلب البيانات وتحليل المرحلة
                    data = await data_fetcher.get_coin_data(coin_data['coingecko_id'])
                    phase_analysis = data['phase_analysis']
                    current_price = data['price']
                    
                    # إرسال إشعار إذا كانت الثقة عالية
                    if phase_analysis['confidence'] > 0.6:
                        await notifier.send_phase_alert(
                            coin_key, 
                            phase_analysis, 
                            current_price
                        )
                    
                    # تسجيل النتائج
                    logger.info(f"{coin_key.upper()}: {phase_analysis['phase']} (ثقة: {phase_analysis['confidence']})")
                    
                    # انتظار بين العملات
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"خطأ في تحليل {coin_key}: {e}")
                    continue
            
            # انتظار 30 دقيقة بين الدورات الكاملة
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"خطأ في مهمة المراقبة: {e}")
            await asyncio.sleep(60)

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "بوت مراقبة مراحل السوق",
        "status": "نشط",
        "version": "6.0.0",
        "feature": "تحليل مراحل السوق (تجميع، صعود، توزيع، هبوط)"
    }

@app.get("/phase/{coin}")
async def get_coin_phase(coin: str):
    """الحصول على مرحلة السوق لعملة محددة"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail="العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data['coingecko_id'])
    
    return {
        "coin": coin,
        "price": data['price'],
        "phase_analysis": data['phase_analysis'],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/alert/{coin}")
async def send_phase_alert(coin: str):
    """إرسال إشعار يدوي بمرحلة السوق"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=400, detail="العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    data = await data_fetcher.get_coin_data(coin_data['coingecko_id'])
    
    success = await notifier.send_phase_alert(coin, data['phase_analysis'], data['price'])
    
    return {
        "message": "تم إرسال الإشعار",
        "success": success,
        "phase": data['phase_analysis']['phase']
    }

@app.get("/status")
async def status():
    """حالة البوت"""
    return {
        "status": "نشط",
        "monitoring": "نشط",
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    }

# بدء المهمة التلقائية
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
