from flask import Flask
import requests
import time
import threading
import logging

app = Flask(__name__)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# الروابط الجديدة
URLS = [
    "https://scanner-iae7.onrender.com",
    "https://applicant-7klo.onrender.com", 
    "https://monitor-19ny.onrender.com",
    "https://ping2-y7lo.onrender.com",
    "https://ping1.onrender.com"
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
        <li>https://scanner-iae7.onrender.com</li>
        <li>https://applicant-7klo.onrender.com</li>
        <li>https://monitor-19ny.onrender.com</li>
        <li>https://ping2-y7lo.onrender.com</li>
        <li>https://ping1.onrender.com</li>
        <li>https://ping-y0gt.onrender.com</li>
    </ul>
    <p>⏰ معدل النبضات: كل 3 دقائق</p>
    <p>🟢 البوت يعمل في الخلفية</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
