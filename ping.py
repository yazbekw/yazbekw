from flask import Flask
import requests
import time
import threading
import logging

app = Flask(__name__)

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# الروابط المحدثة
URLS = [
    "https://ping2-r2ni.onrender.com",
    "https://ping1-5j34.onrender.com",
    "https://monitor-oqk7.onrender.com",
    "https://buy-scanner.onrender.com",
    "https://eth-dhvl.onrender.com",
    "https://scanner-8ika.onrender.com",
]

def send_pings():
    """دالة إرسال النبضات في الخلفية كل 3 دقائق"""
    while True:
        logging.info("🔗 [الكود 1] بدء جولة النبضات (كل 3 دقائق)...")
        
        success_count = 0
        total_count = len(URLS)
        
        for url in URLS:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    logging.info(f"✅ {url} - ناجح")
                    success_count += 1
                else:
                    logging.info(f"⚠️  {url} - حالة: {response.status_code}")
            except Exception as e:
                logging.info(f"❌ {url} - خطأ: {e}")
            
            time.sleep(1)  # انتظار بين الروابط
        
        logging.info(f"📊 [الكود 1] إحصائيات الجولة: {success_count}/{total_count} ناجح")
        logging.info("⏳ [الكود 1] انتظار 3 دقائق للجولة التالية...")
        time.sleep(180)  # 3 دقائق

# بدء النبضات في thread منفصل
ping_thread = threading.Thread(target=send_pings, daemon=True)
ping_thread.start()

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 بوت النبضات 1</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .url-list {
                list-style: none;
                padding: 0;
            }
            .url-list li {
                background: #f8f9fa;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                border-right: 4px solid #667eea;
                transition: all 0.3s ease;
            }
            .url-list li:hover {
                background: #e3f2fd;
                transform: translateX(-5px);
            }
            .status {
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                border-radius: 8px;
                background: #d4edda;
                color: #155724;
                font-weight: bold;
            }
            .info-box {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 بوت النبضات 1 يعمل</h1>
            
            <div class="status">
                🟢 البوت يعمل في الخلفية - إرسال نبضات كل 3 دقائق
            </div>
            
            <div class="info-box">
                <strong>📊 معلومات التشغيل:</strong><br>
                • ⏰ معدل النبضات: كل 3 دقائق<br>
                • 🔗 عدد الروابط: 6 روابط<br>
                • 🎯 الهدف: منع السبات على Render
            </div>
            
            <p><strong>الروابط المستهدفة:</strong></p>
            <ul class="url-list">
                <li>🔗 https://ping2-r2ni.onrender.com</li>
                <li>🔗 https://ping1-5j34.onrender.com</li>
                <li>🔗 https://ping-397j.onrender.com</li>
                <li>🔗 https://scanner-8ika.onrender.com</li>
                <li>🔗 https://applicant-hezk.onrender.com</li>
                <li>🔗 https://monitor-oqk7.onrender.com</li>
            </ul>
            
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>⚡ نظام النبضات التلقائي - الإصدار 1.0</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health_check():
    """نقطة فحص الصحة"""
    return {
        "status": "healthy",
        "service": "ping-bot-1",
        "timestamp": time.time(),
        "urls_count": len(URLS),
        "interval": "180 seconds"
    }

@app.route('/urls')
def get_urls():
    """عرض جميع الروابط المستهدفة"""
    return {
        "urls": URLS,
        "total_count": len(URLS)
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
