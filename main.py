from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import numpy as np
import asyncio
import telegram
import os
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

# تهيئة بوت التلغرام
bot = None
if TELEGRAM_BOT_TOKEN:
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        logger.info("✅ تم تهيئة بوت التلغرام بنجاح")
    except Exception as e:
        logger.error(f"❌ خطأ في تهيئة بوت التلغرام: {e}")
        bot = None

# دالة لجلب بيانات BTC من CoinGecko API
def get_btc_data(days: int = 30) -> pd.DataFrame:
    """
    جلب بيانات تاريخية لـ BTC (سعر، حجم).
    """
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("فشل في جلب البيانات من API")
    
    data = response.json()
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    # دمج السعر والحجم
    df = pd.merge(prices, volumes, on='timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# دالة حساب RSI
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    حساب RSI باستخدام pandas وnumpy.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# دالة حساب MACD
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    حساب MACD البسيط.
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

# دالة حساب Bollinger Bands
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
    """
    حساب Bollinger Bands.
    """
    sma = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    
    upper_band = sma + (rolling_std * std)
    lower_band = sma - (rolling_std * std)
    
    return {
        'sma': sma,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'band_width': (upper_band - lower_band) / sma
    }

# دالة حساب Stochastic Oscillator
def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    حساب Stochastic Oscillator.
    """
    # نظرًا لأننا لا نملك بيانات high/low منفصلة، نستخدم السعر لكل منهم
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    
    return {'k': k_percent, 'd': d_percent}

# دالة حساب OBV (On-Balance Volume)
def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    حساب On-Balance Volume.
    """
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=close.index)

# دالة حساب قوة الإشارة من 1 إلى 10
def calculate_signal_strength(indicator_value: float, buy_threshold: float, sell_threshold: float) -> Tuple[int, str]:
    """
    حساب قوة الإشارة من 1 إلى 10.
    """
    if indicator_value <= buy_threshold:
        # إشارة شراء - كلما كان المؤشر أقل من عتبة الشراء، كانت الإشارة أقوى
        strength = min(10, int((buy_threshold - indicator_value) / buy_threshold * 10) + 1)
        return strength, "شراء"
    elif indicator_value >= sell_threshold:
        # إشارة بيع - كلما كان المؤشر أعلى من عتبة البيع، كانت الإشارة أقوى
        strength = min(10, int((indicator_value - sell_threshold) / (100 - sell_threshold) * 10) + 1)
        return strength, "بيع"
    else:
        return 0, "محايد"

# دالة تحليل جميع المؤشرات
def analyze_all_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    تحليل جميع المؤشرات الفنية وإرجاع النتائج مع قوة الإشارة.
    """
    current_price = df['price'].iloc[-1]
    current_volume = df['volume'].iloc[-1]
    
    # حساب جميع المؤشرات
    rsi = calculate_rsi(df['price']).iloc[-1]
    macd_data = calculate_macd(df['price'])
    macd = macd_data['macd'].iloc[-1]
    signal_line = macd_data['signal'].iloc[-1]
    histogram = macd_data['histogram'].iloc[-1]
    
    # حساب Bollinger Bands
    bb_data = calculate_bollinger_bands(df['price'])
    bb_position = (current_price - bb_data['lower_band'].iloc[-1]) / (bb_data['upper_band'].iloc[-1] - bb_data['lower_band'].iloc[-1]) * 100
    
    # حساب Stochastic (باستخدام السعر كـ high/low/close نظرًا لعدم توفر البيانات)
    stoch_data = calculate_stochastic(df['price'], df['price'], df['price'])
    stoch_k = stoch_data['k'].iloc[-1]
    stoch_d = stoch_data['d'].iloc[-1]
    
    # حساب OBV
    obv = calculate_obv(df['price'], df['volume'])
    obv_trend = "صاعد" if obv.iloc[-1] > obv.iloc[-2] else "هابط"
    
    # حساب قوة الإشارة لكل مؤشر
    rsi_strength, rsi_signal = calculate_signal_strength(rsi, 30, 70)
    macd_strength = 8 if macd > signal_line and histogram > 0 else (8 if macd < signal_line and histogram < 0 else 0)
    macd_signal = "شراء" if macd > signal_line and histogram > 0 else "بيع" if macd < signal_line and histogram < 0 else "محايد"
    
    bb_strength, bb_signal = calculate_signal_strength(bb_position, 20, 80)
    stoch_strength, stoch_signal = calculate_signal_strength(stoch_k, 20, 80)
    
    # حجم التداول
    volume_avg = df['volume'].tail(20).mean()
    volume_ratio = current_volume / volume_avg
    volume_signal = "قوي" if volume_ratio > 1.2 else "ضعيف" if volume_ratio < 0.8 else "عادي"
    
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
            'MACD': {'value': round(macd, 4), 'strength': macd_strength, 'signal': macd_signal},
            'Bollinger_Bands': {'value': round(bb_position, 2), 'strength': bb_strength, 'signal': bb_signal},
            'Stochastic': {'value': round(stoch_k, 2), 'strength': stoch_strength, 'signal': stoch_signal},
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
    if not bot:
        logger.warning("⚠️  بوت التلغرام غير مهيئ - لم يتم إرسال الرسالة")
        return
    
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
        logger.info("✅ تم إرسال الرسالة إلى التلغرام")
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
            df = get_btc_data()
            analysis = analyze_all_indicators(df)
            
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

# Endpoint رئيسي للتنبؤ
@app.get("/")
async def root():
    """
    الصفحة الرئيسية.
    """
    return {
        "message": "مرحباً بك في BTC Trading Bot",
        "status": "يعمل",
        "endpoints": {
            "/predict": "الحصول على تحليل BTC الحالي",
            "/health": "فحص صحة الخادم",
            "/test-telegram": "اختبار إرسال رسالة تلغرام",
            "/start-monitoring": "بدء المراقبة التلقائية"
        },
        "telegram_configured": bool(bot and TELEGRAM_CHAT_ID)
    }

@app.get("/predict")
async def predict_btc() -> JSONResponse:
    """
    يجلب البيانات، يحسب المؤشرات، ويتنبأ بحركة السعر.
    """
    try:
        df = get_btc_data()
        analysis = analyze_all_indicators(df)
        
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"خطأ في التنبؤ: {str(e)}"})

# Endpoint فحص الصحة (لـ Render)
@app.get("/health")
async def health_check() -> JSONResponse:
    """
    فحص صحة البوت: يتحقق من الاتصال بالـ API وحالة الخادم.
    """
    try:
        # اختبار سريع لـ API
        response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            status = {
                "status": "healthy",
                "message": "البوت يعمل بشكل طبيعي",
                "api_status": "up",
                "telegram_configured": bool(bot and TELEGRAM_CHAT_ID),
                "timestamp": datetime.now().isoformat()
            }
            return JSONResponse(content=status)
        else:
            raise Exception("API غير متاح")
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "status": "unhealthy",
            "message": f"مشكلة: {str(e)}",
            "telegram_configured": bool(bot and TELEGRAM_CHAT_ID),
            "timestamp": datetime.now().isoformat()
        })

# Endpoint لبدء الفحص التلقائي
@app.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks) -> JSONResponse:
    """
    بدء المراقبة التلقائية.
    """
    background_tasks.add_task(auto_check)
    return JSONResponse(content={
        "message": "بدأت المراقبة التلقائية كل 30 دقيقة",
        "telegram_configured": bool(bot and TELEGRAM_CHAT_ID)
    })

# Endpoint لإرسال رسالة تجريبية
@app.post("/test-telegram")
async def test_telegram() -> JSONResponse:
    """
    إرسال رسالة تجريبية إلى التلغرام.
    """
    if not bot:
        return JSONResponse(status_code=400, content={"error": "بوت التلغرام غير مهيئ"})
    
    try:
        test_message = "🧪 **رسالة تجريبية**\nهذا اختبار لنظام إشعارات BTC Bot\n✅ البوت يعمل بشكل صحيح\n⏰ الوقت: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await send_telegram_message(test_message)
        return JSONResponse(content={"message": "تم إرسال الرسالة التجريبية"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"خطأ في إرسال الرسالة: {str(e)}"})

# بدء الفحص التلقائي عند تشغيل التطبيق
@app.on_event("startup")
async def startup_event():
    """
    بدء الفحص التلقائي عند تشغيل التطبيق.
    """
    if bot and TELEGRAM_CHAT_ID:
        logger.info("🚀 بدء الفحص التلقائي عند التشغيل...")
        asyncio.create_task(auto_check())
    else:
        logger.warning("⚠️  الفحص التلقائي متوقف - مفاتيح التلغرام غير محددة")

# تشغيل الخادم (للتطوير المحلي)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
