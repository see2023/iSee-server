from flask import Flask, request, jsonify, make_response
import logging
import os
import random
from livekit import api
import datetime
import base64

from auth import authenticate_request
from live import live_bp
logger = logging.getLogger()
from common.config import config

test_room = 'my-room'

def getToken(room_name, id, name):
  token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
    .with_identity(id) \
    .with_name(name) \
    .with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
    ))
  return token.to_jwt()

@live_bp.route('/getToken', methods=['POST'])
@authenticate_request
def post_message():
    data = request.get_json()
    room_name = data.get('room')
    if room_name is None or room_name == '':
        room_name = test_room
    id = data.get('user')
    if id is None or id == '':
        id = 'user' + str(random.randint(1, 100000))
    name = data.get('name')
    if name is None or name == '':
        name = id
    logger.info(f'Received token request of room: {room_name} from user: {data["user"]}')
    token = getToken(room_name, id, name)
    response = {'status': True, 'text': token}
    return jsonify(response)


@live_bp.route('/putImg', methods=['POST'])
@authenticate_request
def put_img():
    data = request.get_json()
    img_content_str: str = data.get('text')
    if img_content_str is None or img_content_str == '':
        return make_response({'status': False, 'text': '图片内容为空'}, 400)
    # base64解码
    img_content = base64.b64decode(img_content_str)

    # 生成目录 年月日
    now = datetime.datetime.now()
    file_path = now.strftime('%Y%m%d')
    full_dir = os.path.join(config.api.www_root, file_path )
    file_name = now.strftime('%H%M%S%f') + '_' + str(random.randint(1000000, 9000000)) + '.jpg'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    # 保存图片到本地
    with open(os.path.join(full_dir, file_name), 'wb') as f:
        f.write(img_content)
    response = {'status': True, 'text':  file_path + '/' + file_name, "size": len(img_content), "str_size": len(img_content_str), "first_10": img_content_str[:10]}
    return jsonify(response)
