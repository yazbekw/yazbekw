from flask import Flask
import requests
import time
import threading
import logging

app = Flask(__name__)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# الروابط بدون الرابط الذاتي
URLS = [
    "https://yazbek-1-ozjd.onrender.com",
    "https://yazbek-3.onrender.com",
    "https://scanner-zwlt.onrender.com",
    "https://testnet-7t23.onrender.com",
    "https://trade-manager-j8ur.onrender.com",
    "https://trade-hbwj.onrender.com"
]

def send_pings():
    """دالة إرسال النبضات في الخلفية كل 3 دقائق"""
    while True:
        logging.info("🔗 [الكود 1] بدء جولة النبضات (كل 3 دقائق)...")
        
        for url in URLS:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    logging.info(f"✅ {url} - ناجح")
                else:
                    logging.info(f"⚠️  {url} - حالة: {response.status_code}")
            except Exception as e:
                logging.info(f"❌ {url} - خطأ: {e}")
            
            time.sleep(1)  # انتظار بين الروابط
        
        logging.info("⏳ [الكود 1] انتظار 3 دقائق للجولة التالية...")
        time.sleep(180)  # 3 دقائق

# بدء النبضات في thread منفصل
ping_thread = threading.Thread(target=send_pings, daemon=True)
ping_thread.start()

@app.route('/')
def home():
    return """
    <h1>🚀 بوت النبضات 1 يعمل</h1>
    <p>إرسال نبضات كل 3 دقائق إلى:</p>
    <ul>
        <li>https://yazbek-2-482e.onrender.com</li>
        <li>https://yazbek-3.onrender.com</li>
        <li>https://crypto-scalping.onrender.com</li>
    </ul>
    <p>⏰ معدل النبضات: كل 3 دقائق</p>
    <p>🟢 البوت يعمل في الخلفية</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
