from auth import gen_post_request
import aiohttp
from typing import Optional
import os
import logging
from PIL import  Image
import base64
from common.config import config



async def common_post(url, user, text):
    post_data = await gen_post_request(user, text)
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=post_data) as response:
            if response.status != 200:
                print('http post failed, url: ', url) 
                return None
            else:
                return await response.json()

async def get_token(user):
    try:
        url = os.getenv("LIVEKIT_API_URL") + '/api/v1/live/getToken'
        res =  await common_post(url, user, '')
        if not res or not res['status']:
            return None
        return res['text']
    except Exception as e:
        logging.error(f"Failed to get token for user: {user}, error: {e}")
        return None


async def upload_img(user: str, img: Image) -> Optional[str]:
    try:
        url = os.getenv("LIVEKIT_API_URL") + '/api/v1/live/putImg'
        img_data = img.tobytes()
        base64_img = base64.b64encode(img_data).decode('utf-8')
        post_data = gen_post_request(user, base64_img)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=post_data) as response:
                if response.status != 200:
                    logging.error(f"Failed to upload img for user: {user}, status_code: {response.status}")
                    return None
                else:
                    res = await response.json()
                    url = res.get('text')
                    if url:
                        return config.api.url_prefix + url
                    else:
                        return None
    except Exception as e:
        logging.error(f"Failed to upload img for user: {user}, error: {e}")
        return None
