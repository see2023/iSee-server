import redis
import os
import logging
import time

REDIS_PREFIX = 'isee:'
REDIS_CHAT_KEY = REDIS_PREFIX + 'chat:'
REDIS_MQ_KEY = REDIS_PREFIX +'mq'
REDIS_DETECTED_NAMES_KEY = REDIS_PREFIX +'detected_names'

class RedisClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise Exception('This is a singleton class, use get_instance() instead')
        self.client = redis.Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'), db=0, decode_responses=True, 
                                # ssl_ca_certs=os.getenv('REDIS_CAFILE'), ssl_certfile=os.getenv('REDIS_CERTFILE'), 
                                password=os.getenv('REDIS_PASSWORD'), ssl=True, ssl_cert_reqs='none', 
                                socket_timeout=10, health_check_interval=30, retry_on_timeout=True, 
                        )

def get_redis_client() -> redis.Redis:
    redis_client = RedisClient.get_instance()
    return redis_client.client

def write_chat_to_redis(key, text: str, srcname: str, timestamp: float, duration: float = 0, language: str = 'zh', combine_same_srcname: bool = False):
    redis_client = RedisClient.get_instance()
    # 获取最新的一个消息，如果srcname和当前相同，则删除刚才的消息，将消息内容合并到本消息中
    # keep alive

    try:
        redis_client.client.ping()
        if combine_same_srcname:
            latest_msg = redis_client.client.xrevrange(key, max= '+', count=1)
            if latest_msg and len(latest_msg)>0 and latest_msg[0][1]['srcname'] == srcname:
                last_duration = latest_msg[0][1]['duration']
                if last_duration is None:
                    last_duration = 0
                else:
                    last_duration = float(last_duration)
                duration = duration + last_duration
                text = latest_msg[0][1]['text'] + text
                lasttime = int(latest_msg[0][1]['timestamp'])
                if timestamp - lasttime/1000 <  600:
                    logging.info('同类消息间隔小于600秒，合并消息')
                    redis_client.client.xdel(key, latest_msg[0][0])
                    redis_client.client.xadd(key, {"text": text, "timestamp": lasttime, "duration": duration, "language": language, "srcname": srcname})
                    logging.info(f"write_chat_to_redis combined: {text} {srcname} {timestamp} {duration} {language}")
                    return
        redis_client.client.xadd(key, {"text": text, "timestamp": int(timestamp*1000), "duration": duration, "language": language, "srcname": srcname})
        logging.debug(f"write_chat_to_redis: {text} {srcname} {timestamp} {duration} {language}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 
    
def write_mq_to_redis(message: str):
    redis_client = RedisClient.get_instance()
    try:
        redis_client.client.ping()
        redis_client.client.publish(REDIS_MQ_KEY, message)
        logging.info(f"write_mq_to_redis: {message}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 

def write_detected_names_to_redis(names: str):
    redis_client = RedisClient.get_instance()
    try:
        redis_client.client.set(REDIS_DETECTED_NAMES_KEY, names, ex=10)
        logging.info(f"write_detected_names_to_redis: {names}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 

def get_detected_names_from_redis():
    redis_client = RedisClient.get_instance()
    try:
        msgs = redis_client.client.get(REDIS_DETECTED_NAMES_KEY)
        if msgs is None:
            return ""
        return msgs
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
        return ""
    except Exception as e:
        logging.error(f"Redis error: {e}") 
        return ""
