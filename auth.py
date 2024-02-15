from flask import Flask, request, jsonify, make_response
from redis import Redis
import hashlib
import logging
from datetime import datetime
from db.redis_cli import get_redis_client, REDIS_PREFIX

logger = logging.getLogger()
redis_client = get_redis_client()

def authenticate_request(func):
    def wrapper(*args, **kwargs):
        data = request.get_json()
        if not data or 'user' not in data or 'timestamp' not in data or 'text' not in data or 'md5hash' not in data:
            logger.error('Invalid request')
            response = {'status': False, 'text': 'Invalid request'}
            return make_response(jsonify(response), 400)

        user_key = redis_client.get(REDIS_PREFIX + data['user'])
        if not user_key:
            logger.error(f"User not found: {data['user']}")
            response = {'status': False, 'text': 'user not found'}
            return make_response(jsonify(response), 401)

        raw_string = f"{data['user']}{user_key}{data['timestamp']}{data['text']}".encode()
        md5hash = hashlib.md5(raw_string).hexdigest()

        if md5hash == data['md5hash']:
            return func(*args, **kwargs)
        else:
            logger.error(f"Authentication failed for user: {data['user']}")
            response = {'status': False, 'text': 'authentication failed'}
            return make_response(jsonify(response), 401)

    wrapper.__name__ = func.__name__
    return wrapper

def gen_md5hash(user, timestamp, text):
    # 获取存储于Redis的密钥
    user_key = redis_client.get(REDIS_PREFIX + user)
    if not user_key:
        logger.error(f"User not found: {user}")
        return ''

    # 生成MD5散列
    raw_string = f"{user}{user_key}{timestamp}{text}".encode()
    md5hash = hashlib.md5(raw_string).hexdigest()
    return md5hash

def gen_post_request(user, text):
    timestamp = datetime.now().timestamp()
    md5hash = gen_md5hash(user, timestamp, text)
    if not md5hash:
        return None

    data = {
        'user': user,
        'timestamp': timestamp,
        'text': text,
        'md5hash': md5hash
    }
    return data

