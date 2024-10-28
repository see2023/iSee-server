import redis
import os
import logging
import socket

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
        self.client = redis.asyncio.Redis(
            host=os.getenv('REDIS_HOST'), 
            port=os.getenv('REDIS_PORT'), 
            db=0, 
            decode_responses=True, 
            socket_keepalive=True,
            password=os.getenv('REDIS_PASSWORD'), 
            ssl=True, 
            ssl_cert_reqs='none',
            socket_timeout=60,
            health_check_interval=30,
            retry_on_timeout=True,
            socket_keepalive_options={
                socket.TCP_KEEPALIVE: 30,  
                socket.TCP_KEEPINTVL: 10, 
                socket.TCP_KEEPCNT: 3     
            }
        )

def get_redis_client() -> redis.asyncio.Redis:
    redis_client = RedisClient.get_instance()
    return redis_client.client

async def write_chat_to_redis(key, text: str, srcname: str, timestamp: float, duration: float = 0, language: str = 'zh', combine_same_srcname: bool = False):
    client: redis.asyncio.Redis = get_redis_client()

    try:
        if combine_same_srcname:
            latest_msg = await client.xrevrange(key, max= '+', count=1)
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
                    await client.xdel(key, latest_msg[0][0])
                    await client.xadd(key, {"text": text, "timestamp": lasttime, "duration": duration, "language": language, "srcname": srcname})
                    logging.info(f"write_chat_to_redis combined: {text} {srcname} {timestamp} {duration} {language}")
                    return
        rt = await client.xadd(key, {"text": text, "timestamp": int(timestamp*1000), "duration": duration, "language": language, "srcname": srcname})
        logging.debug(f"write_chat_to_redis: {text} {srcname} {timestamp} {duration} {language}, return: {rt}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 
    
async def write_mq_to_redis(message: str):
    client = get_redis_client()
    try:
        await client.publish(REDIS_MQ_KEY, message)
        logging.info(f"write_mq_to_redis: {message}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 

async def write_detected_names_to_redis(names: str):
    client = get_redis_client()
    try:
        await client.set(REDIS_DETECTED_NAMES_KEY, names, ex=10)
        logging.info(f"write_detected_names_to_redis: {names}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}") 

async def get_detected_names_from_redis():
    client = get_redis_client()
    try:
        msgs = await client.get(REDIS_DETECTED_NAMES_KEY)
        if msgs is None:
            return ""
        return msgs
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
        return ""
    except Exception as e:
        logging.error(f"Redis error: {e}") 
        return ""

async def write_summary_and_state_to_redis(user_id: str, summary: str, user_state: str):
    client = get_redis_client()
    try:
        await client.set(f"{REDIS_CHAT_KEY}{user_id}:summary", summary)
        await client.set(f"{REDIS_CHAT_KEY}{user_id}:user_state", user_state)
        logging.info(f"Wrote summary and user_state to Redis for user {user_id}")
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
    except Exception as e:
        logging.error(f"Redis error: {e}")

async def get_summary_and_state_from_redis(user_id: str):
    client = get_redis_client()
    try:
        summary = await client.get(f"{REDIS_CHAT_KEY}{user_id}:summary")
        user_state = await client.get(f"{REDIS_CHAT_KEY}{user_id}:user_state")
        logging.info(f"Read summary and user_state from Redis for user {user_id}")
        if not summary:
            summary = ""
        if not user_state:
            user_state = ""
        return summary, user_state
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Redis error: {e}")
        return None, None
