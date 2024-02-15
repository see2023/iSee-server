from flask import Flask, request, jsonify, make_response
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
if __name__ == '__main__':
    app.config['SERVER_NAME'] = None
else:
    app.config['SERVER_NAME'] = 'localhost:5001'


from live import live_bp
app.register_blueprint(live_bp)


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
)
logger = logging.getLogger()
# handler = RotatingFileHandler('api.log', maxBytes=5000000, backupCount=1)
# logger.addHandler(handler)



@app.route('/', methods=['POST', 'GET'])
def hello():
    response = {'status': True, 'text': 'Hello from Flask!', 'timestamp': datetime.now().isoformat()}
    return jsonify(response)


# 启动服务器
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

