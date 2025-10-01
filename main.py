from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import numpy as np
import asyncio
import os
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import time
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced BTC Trading Bot", description="Tracks BTC with multiple indicators and Telegram notifications.")

# إعدادات التلغرام - أخذ المفاتيح من متغيرات البيئة
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# التحقق من وجود المفاتيح
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("⚠️  مفاتيح التلغرام غير محددة في متغيرات البيئة")

# دالة لجلب بيانات BTC من CoinGecko API
def get_btc_data(days: int = 30) -> Dict[str, Any]:
    """
    جلب بيانات تاريخية لـ BTC (سعر، حجم).
    """
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("فشل في جلب البيانات من API")
    
    data = response.json()
    return data

# دالة حساب المتوسط المتحرك
def calculate_ema(prices: List[float], period: int) -> List[float]:
    """
    حساب المتوسط المتحرك الأسي.
    """
    if len(prices) < period:
        return [0] * len(prices)
    
    ema = [prices[0]]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema

# دالة حساب RSI بدون pandas
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    حساب RSI بدون استخدام pandas.
    """
    if len(prices) < period + 1:
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
    
    # استخدام آخر 'period' من البيانات
    recent_gains = gains[-period:]
    recent_losses = losses[-period:]
    
    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# دالة حساب MACD بدون pandas
def calculate_macd(prices: List[float]) -> Dict[str, float]:
    """
    حساب MACD بدون استخدام pandas.
    """
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    
    macd_line = ema_12[-1] - ema_26[-1]
    
    # حساب خط الإشارة (EMA 9 لـ MACD)
    macd_values = [ema_12[i] - ema_26[i] for i in range(len(prices))]
    signal_line = calculate_ema(macd_values, 9)[-1]
    
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

# دالة حساب Bollinger Bands بدون pandas
def calculate_bollinger_bands(prices: List[float], period: int = 20, std: int = 2) -> Dict[str, float]:
    """
    حساب Bollinger Bands بدون استخدام pandas.
    """
    if len(prices) < period:
        recent_prices = prices
    else:
        recent_prices = prices[-period:]
    
    sma = sum(recent_prices) / len(recent_prices)
    
    variance = sum((x - sma) ** 2 for x in recent_prices) / len(recent_prices)
    std_dev = variance ** 0.5
    
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    
    current_price = prices[-1]
    bb_position = ((current_price - lower_band) / (upper_band - lower_band)) * 100
    
    return {
        'sma': sma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'position': bb_position
    }

# دالة حساب Stochastic Oscillator بدون pandas
def calculate_stochastic(prices: List[float], period: int = 14) -> Dict[str, float]:
    """
    حساب Stochastic Oscillator بدون استخدام pandas.
    """
    if len(prices) < period:
        recent_prices = prices
    else:
        recent_prices = prices[-period:]
    
    highest_high = max(recent_prices)
    lowest_low = min(recent_prices)
    current_close = prices[-1]
    
    if highest_high == lowest_low:
        k_percent = 50
    else:
        k_percent = 100 * ((current_close - lowest_low) / (highest_high - lowest_low))
    
    # حساب %D (المتوسط المتحرك لـ %K)
    k_values = []
    for i in range(len(prices) - period + 1):
        period_high = max(prices[i:i+period])
        period_low = min(prices[i:i+period])
        period_close = prices[i+period-1]
        
        if period_high == period_low:
            k_val = 50
        else:
            k_val = 100 * ((period_close - period_low) / (period_high - period_low))
        k_values.append(k_val)
    
    d_percent = sum(k_values[-3:]) / min(3, len(k_values)) if k_values else 50
    
    return {'k': k_percent, 'd': d_percent}

# دالة حساب قوة الإشارة من 1 إلى 10
def calculate_signal_strength(indicator_value: float, buy_threshold: float, sell_threshold: float) -> Tuple[int, str]:
    """
    حساب قوة الإشارة من 1 إلى 10.
    """
    if indicator_value <= buy_threshold:
        strength = min(10, int((buy_threshold - indicator_value) / buy_threshold * 10) + 1)
        return strength, "شراء"
    elif indicator_value >= sell_threshold:
        strength = min(10, int((indicator_value - sell_threshold) / (100 - sell_threshold) * 10) + 1)
        return strength, "بيع"
    else:
        return 0, "محايد"

# دالة تحليل جميع المؤشرات
def analyze_all_indicators(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    تحليل جميع المؤشرات الفنية وإرجاع النتائج مع قوة الإشارة.
    """
    prices = [point[1] for point in data['prices']]
    volumes = [point[1] for point in data['total_volumes']]
    
    current_price = prices[-1]
    current_volume = volumes[-1]
    
    # حساب جميع المؤشرات
    rsi = calculate_rsi(prices)
    macd_data = calculate_macd(prices)
    bb_data = calculate_bollinger_bands(prices)
    stoch_data = calculate_stochastic(prices)
    
    # حساب قوة الإشارة لكل مؤشر
    rsi_strength, rsi_signal = calculate_signal_strength(rsi, 30, 70)
    
    macd_strength = 8 if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0 else (8 if macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0 else 0)
    macd_signal = "شراء" if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0 else "بيع" if macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0 else "محايد"
    
    bb_strength, bb_signal = calculate_signal_strength(bb_data['position'], 20, 80)
    stoch_strength, stoch_signal = calculate_signal_strength(stoch_data['k'], 20, 80)
    
    # حجم التداول
    volume_avg = sum(volumes[-20:]) / min(20, len(volumes))
    volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
    volume_signal = "قوي" if volume_ratio > 1.2 else "ضعيف" if volume_ratio < 0.8 else "عادي"
    
    # اتجاه OBV مبسط
    obv_trend = "صاعد" if current_volume > volumes[-2] if len(volumes) > 1 else current_volume else "هابط"
    
    # إشارة عامة مجمعة
    buy_signals = sum([rsi_strength if rsi_signal == "شراء" else 0,
                      macd_strength if macd_signal == "شراء" else 0,
                      bb_strength if bb_signal == "شراء" else 0,
                      stoch_strength if stoch_signal == "شراء" else 0])
    
    sell_signals = sum([rsi_strength if rsi_signal == "بيع" else 0,
                       macd_strength if macd_signal == "بيع" else 0,
                       bb_strength if bb_signal == "بيع" else 0,
                       stoch_strength if stoch_signal == "بيع" else 0])
    
    if buy_signals > sell_signals:
        overall_signal = f"شراء (قوة: {min(10, buy_signals//4)})"
    elif sell_signals > buy_signals:
        overall_signal = f"بيع (قوة: {min(10, sell_signals//4)})"
    else:
        overall_signal = "محايد"
    
    return {
        'timestamp': datetime.now(),
        'current_price': round(current_price, 2),
        'indicators': {
            'RSI': {'value': round(rsi, 2), 'strength': rsi_strength, 'signal': rsi_signal},
            'MACD': {'value': round(macd_data['macd'], 4), 'strength': macd_strength, 'signal': macd_signal},
            'Bollinger_Bands': {'value': round(bb_data['position'], 2), 'strength': bb_strength, 'signal': bb_signal},
            'Stochastic': {'value': round(stoch_data['k'], 2), 'strength': stoch_strength, 'signal': stoch_signal},
            'Volume': {'value': round(volume_ratio, 2), 'signal': volume_signal},
            'OBV': {'trend': obv_trend}
        },
        'overall_signal': overall_signal
    }

# دالة إرسال رسالة تلغرام
async def send_telegram_message(message: str):
    """
    إرسال رسالة إلى قناة/مجموعة التلغرام.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️  بوت التلغرام غير مهيئ - لم يتم إرسال الرسالة")
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info("✅ تم إرسال الرسالة إلى التلغرام")
        else:
            logger.error(f"❌ خطأ في إرسال الرسالة: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال الرسالة إلى التلغرام: {e}")

# دالة الفحص التلقائي
async def auto_check():
    """
    فحص تلقائي للمؤشرات كل 30 دقيقة وإرسال إشعارات.
    """
    while True:
        try:
            logger.info("🔄 بدء الفحص التلقائي...")
            data = get_btc_data()
            analysis = analyze_all_indicators(data)
            
            # إنشاء رسالة التلغرام
            message = f"📊 **تقرير تحليل BTC**\n"
            message += f"⏰ الوقت: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"💰 السعر الحالي: ${analysis['current_price']:,.2f}\n\n"
            message += f"**المؤشرات:**\n"
            
            for indicator, data in analysis['indicators'].items():
                if 'strength' in data:
                    strength_stars = "⭐" * data['strength']
                    message += f"• {indicator}: {data['value']} | {data['signal']} | قوة: {data['strength']}/10 {strength_stars}\n"
                else:
                    message += f"• {indicator}: {data.get('value', data.get('trend', 'N/A'))}\n"
            
            message += f"\n**الإشارة العامة: {analysis['overall_signal']}**\n"
            message += f"\n⚠️ تنبيه: هذا تحليل فني وليس نصيحة استثمارية"
            
            # إرسال الرسالة
            await send_telegram_message(message)
            
            logger.info(f"✅ تم الانتهاء من الفحص التلقائي - الإشارة: {analysis['overall_signal']}")
            
        except Exception as e:
            error_message = f"❌ خطأ في الفحص التلقائي: {str(e)}"
            logger.error(error_message)
            await send_telegram_message(error_message)
        
        # الانتظار 30 دقيقة (1800 ثانية) قبل الفحص التالي
        logger.info("⏰ انتظار 30 دقيقة للفحص التالي...")
        await asyncio.sleep(1800)

# Endpoints الأساسية (نفس الكود السابق)
@app.get("/")
async def root():
    return {
        "message": "مرحباً بك في BTC Trading Bot",
        "status": "يعمل",
        "endpoints": {
            "/predict": "الحصول على تحليل BTC الحالي",
            "/health": "فحص صحة الخادم",
            "/test-telegram": "اختبار إرسال رسالة تلغرام"
        },
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    }

@app.get("/predict")
async def predict_btc() -> JSONResponse:
    try:
        data = get_btc_data()
        analysis = analyze_all_indicators(data)
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"خطأ في التنبؤ: {str(e)}"})

@app.get("/health")
async def health_check() -> JSONResponse:
    try:
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            return JSONResponse(content={
                "status": "healthy",
                "message": "البوت يعمل بشكل طبيعي",
                "api_status": "up",
                "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise Exception("API غير متاح")
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "status": "unhealthy",
            "message": f"مشكلة: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })

@app.post("/test-telegram")
async def test_telegram() -> JSONResponse:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return JSONResponse(status_code=400, content={"error": "بوت التلغرام غير مهيئ"})
    
    try:
        test_message = "🧪 **رسالة تجريبية**\nهذا اختبار لنظام إشعارات BTC Bot\n✅ البوت يعمل بشكل صحيح\n⏰ الوقت: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await send_telegram_message(test_message)
        return JSONResponse(content={"message": "تم إرسال الرسالة التجريبية"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"خطأ في إرسال الرسالة: {str(e)}"})

@app.on_event("startup")
async def startup_event():
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        logger.info("🚀 بدء الفحص التلقائي عند التشغيل...")
        asyncio.create_task(auto_check())
    else:
        logger.warning("⚠️  الفحص التلقائي متوقف - مفاتيح التلغرام غير محددة")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
