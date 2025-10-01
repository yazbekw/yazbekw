from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import asyncio
import os
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
import time
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BTC Trading Bot", description="Advanced BTC trading analysis with Telegram notifications")

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# دالة لجلب بيانات BTC
def get_btc_data(days: int = 30) -> Dict[str, Any]:
    """جلب بيانات BTC من CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"فشل في جلب البيانات: {e}")

# دالة حساب المتوسط المتحرك
def calculate_sma(prices: List[float], period: int) -> float:
    """حساب المتوسط المتحرك البسيط"""
    if len(prices) < period:
        return sum(prices) / len(prices)
    return sum(prices[-period:]) / period

# دالة حساب RSI
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """حساب مؤشر RSI"""
    if len(prices) <= period:
        return 50
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    # استخدام آخر فترة من البيانات
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# دالة حساب MACD
def calculate_macd(prices: List[float]) -> Dict[str, float]:
    """حساب مؤشر MACD"""
    def calculate_ema(data: List[float], period: int) -> List[float]:
        if not data:
            return []
        multiplier = 2 / (period + 1)
        ema = [data[0]]
        for i in range(1, len(data)):
            ema_value = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_value)
        return ema
    
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    
    if not ema_12 or not ema_26:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    macd_line = ema_12[-1] - ema_26[-1]
    
    # حساب خط الإشارة (EMA 9 لـ MACD)
    macd_history = [ema_12[i] - ema_26[i] for i in range(min(len(ema_12), len(ema_26)))]
    signal_line = calculate_ema(macd_history, 9)[-1] if len(macd_history) >= 9 else macd_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': macd_line - signal_line
    }

# دالة حساب Bollinger Bands
def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict[str, float]:
    """حساب Bollinger Bands"""
    if len(prices) < period:
        sma = sum(prices) / len(prices)
        std_dev = 0
    else:
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        variance = sum((x - sma) ** 2 for x in recent_prices) / period
        std_dev = variance ** 0.5
    
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    current_price = prices[-1]
    
    if upper_band != lower_band:
        position = ((current_price - lower_band) / (upper_band - lower_band)) * 100
    else:
        position = 50
    
    return {
        'sma': sma,
        'upper': upper_band,
        'lower': lower_band,
        'position': position
    }

# دالة حساب قوة الإشارة
def calculate_signal_strength(value: float, buy_thresh: float, sell_thresh: float) -> Tuple[int, str]:
    """حساب قوة الإشارة من 1 إلى 10"""
    if value <= buy_thresh:
        strength = min(10, int((buy_thresh - value) / buy_thresh * 10) + 1)
        return strength, "شراء"
    elif value >= sell_thresh:
        strength = min(10, int((value - sell_thresh) / (100 - sell_thresh) * 10) + 1)
        return strength, "بيع"
    else:
        return 0, "محايد"

# التحليل الشامل
def analyze_btc() -> Dict[str, Any]:
    """تحليل شامل لـ BTC باستخدام جميع المؤشرات"""
    try:
        data = get_btc_data(60)  # 60 يوم للحصول على بيانات كافية
        prices = [point[1] for point in data['prices']]
        volumes = [point[1] for point in data['total_volumes']]
        
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        # حساب المؤشرات
        rsi = calculate_rsi(prices)
        macd_data = calculate_macd(prices)
        bb_data = calculate_bollinger_bands(prices)
        
        # حساب قوة الإشارات
        rsi_strength, rsi_signal = calculate_signal_strength(rsi, 30, 70)
        
        macd_signal = "شراء" if macd_data['histogram'] > 0 else "بيع" if macd_data['histogram'] < 0 else "محايد"
        macd_strength = 8 if macd_data['histogram'] > 0 else (8 if macd_data['histogram'] < 0 else 0)
        
        bb_strength, bb_signal = calculate_signal_strength(bb_data['position'], 20, 80)
        
        # تحليل الحجم
        volume_avg = calculate_sma(volumes, 20)
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        volume_signal = "قوي" if volume_ratio > 1.2 else "ضعيف" if volume_ratio < 0.8 else "عادي"
        
        # الإشارة العامة
        buy_signals = sum([
            rsi_strength if rsi_signal == "شراء" else 0,
            macd_strength if macd_signal == "شراء" else 0,
            bb_strength if bb_signal == "شراء" else 0
        ])
        
        sell_signals = sum([
            rsi_strength if rsi_signal == "بيع" else 0,
            macd_strength if macd_signal == "بيع" else 0,
            bb_strength if bb_signal == "بيع" else 0
        ])
        
        if buy_signals > sell_signals:
            overall = f"شراء (قوة: {min(10, buy_signals//3)})"
        elif sell_signals > buy_signals:
            overall = f"بيع (قوة: {min(10, sell_signals//3)})"
        else:
            overall = "محايد"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'price': round(current_price, 2),
            'indicators': {
                'RSI': {'value': round(rsi, 2), 'strength': rsi_strength, 'signal': rsi_signal},
                'MACD': {'value': round(macd_data['macd'], 4), 'strength': macd_strength, 'signal': macd_signal},
                'Bollinger_Bands': {'value': round(bb_data['position'], 2), 'strength': bb_strength, 'signal': bb_signal},
                'Volume': {'value': round(volume_ratio, 2), 'signal': volume_signal}
            },
            'overall_signal': overall
        }
        
    except Exception as e:
        logger.error(f"خطأ في التحليل: {e}")
        raise

# إرسال رسالة تلغرام
async def send_telegram(message: str):
    """إرسال رسالة إلى تلغرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("مفاتيح التلغرام غير متوفرة")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("تم إرسال الرسالة إلى التلغرام")
        else:
            logger.error(f"خطأ في إرسال الرسالة: {response.status_code}")
    except Exception as e:
        logger.error(f"خطأ في إرسال التلغرام: {e}")

# الفحص التلقائي
async def auto_monitor():
    """مراقبة تلقائية كل 30 دقيقة"""
    while True:
        try:
            logger.info("بدء الفحص التلقائي...")
            analysis = analyze_btc()
            
            message = f"📊 **تقرير BTC التلقائي**\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            message += f"💰 السعر: ${analysis['price']:,.2f}\n\n"
            message += "**المؤشرات:**\n"
            
            for name, indicator in analysis['indicators'].items():
                if 'strength' in indicator:
                    stars = "⭐" * indicator['strength']
                    message += f"• {name}: {indicator['value']} | {indicator['signal']} | قوة: {indicator['strength']}/10 {stars}\n"
                else:
                    message += f"• {name}: {indicator['value']} | {indicator['signal']}\n"
            
            message += f"\n**الإشارة: {analysis['overall_signal']}**\n"
            message += "\n⚠️ تحليل فني فقط"
            
            await send_telegram(message)
            logger.info(f"تم إرسال التقرير - {analysis['overall_signal']}")
            
        except Exception as e:
            error_msg = f"❌ خطأ في المراقبة: {str(e)}"
            logger.error(error_msg)
            await send_telegram(error_msg)
        
        await asyncio.sleep(1800)  # 30 دقيقة

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "مرحباً في BTC Trading Bot",
        "status": "نشط",
        "monitoring": "مفعل كل 30 دقيقة" if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else "غير مفعل",
        "endpoints": ["/analysis", "/health", "/test-telegram"]
    }

@app.get("/analysis")
async def get_analysis():
    """الحصول على التحليل الحالي"""
    try:
        analysis = analyze_btc()
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في التحليل: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """فحص صحة الخادم"""
    try:
        # اختبار اتصال API
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        api_status = "up" if response.status_code == 200 else "down"
        
        return {
            "status": "healthy",
            "api_status": api_status,
            "timestamp": datetime.now().isoformat(),
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/test-telegram")
async def test_telegram():
    """اختبار إرسال رسالة تلغرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return JSONResponse(
            status_code=400,
            content={"error": "مفاتيح التلغرام غير محددة"}
        )
    
    try:
        test_msg = "🧪 **اختبار البوت**\n✅ البوت يعمل بشكل صحيح\n⏰ " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await send_telegram(test_msg)
        return {"message": "تم إرسال الرسالة التجريبية"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"فشل في الإرسال: {str(e)}"}
        )

# بدء المراقبة التلقائية
@app.on_event("startup")
async def start_monitoring():
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        logger.info("بدء المراقبة التلقائية...")
        asyncio.create_task(auto_monitor())
    else:
        logger.warning("المراقبة التلقائية متوقفة - مفاتيح التلغرام غير محددة")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
