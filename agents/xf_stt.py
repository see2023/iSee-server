from typing import Optional
from livekit import agents, rtc
from livekit.agents.utils import AudioBuffer
from livekit.agents import stt
import logging
import time
import asyncio

import websockets
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import os
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
from stt_base import MySpeechData
from speaker import Speaker

STATUS_FIRST_FRAME = 0 
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        if not self.APPID or not self.APIKey or not self.APISecret:
            raise ValueError("xunfei APPID, APIKey and APISecret must be set in .env file")

        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1,"vad_eos":10000}

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        url = url + '?' + urlencode(v)
        return url


class STT(stt.STT):
    def __init__(self, *, streaming_supported: bool = False, speaker_detector: Speaker = None) -> None:
        super().__init__(streaming_supported=streaming_supported)
        self.wsParam = Ws_Param(os.getenv('XF_APPID'), os.getenv('XF_API_KEY'), os.getenv('XF_API_SECRET'))
        self.speaker_detector:Speaker = speaker_detector

    def parse_message(self, message) -> str:
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            result = ""
            if code != 0:
                errMsg = json.loads(message)["message"]
                logging.error("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
                return ""
            else:
                data = json.loads(message)["data"]["result"]["ws"]
                # logging.debug(data)
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                logging.debug("sid:%s call success, result is:%s" % (sid, result))
                return result
        except Exception as e:
            logging.error("xunfei receive msg,but parse exception:", e)
            return ""
    
    @classmethod
    def change_sample_rate(cls, buffer: AudioBuffer, sample_rate: int) -> AudioBuffer:
        if buffer.sample_rate == sample_rate:
            return buffer
        return buffer.remix_and_resample(sample_rate, buffer.num_channels)

    async def send_audio(self, websocket, buffer: AudioBuffer) -> None:
        frameSize = 8000  # 每一帧的音频大小
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧
        format = "audio/L16;rate=%d" % buffer.sample_rate
        data_size = len(buffer.data)
        sent_size = 0
        logging.info("xf send audio, data size:%d, sample rate:%d" % (data_size, buffer.sample_rate))

        while True:
            try:
                if status == STATUS_FIRST_FRAME:
                    if frameSize >= data_size:
                        status = STATUS_LAST_FRAME
                        sent_size = data_size
                    else:
                        status = STATUS_CONTINUE_FRAME
                        sent_size = frameSize
                    buf = buffer.data[:sent_size]
                    d = {"common": self.wsParam.CommonArgs,
                            "business": self.wsParam.BusinessArgs,
                            "data": {"status": 0, "format": format,
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"}}
                    d = json.dumps(d)
                    await websocket.send(d)
                elif status == STATUS_CONTINUE_FRAME:
                    if sent_size + frameSize >= data_size:
                        status = STATUS_LAST_FRAME
                        sent_size = data_size
                    else:
                        status = STATUS_CONTINUE_FRAME
                        sent_size += frameSize
                    buf = buffer.data[sent_size-frameSize:sent_size]
                    d = {"data": {"status": 1, "format": format,
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"}}
                    d = json.dumps(d)
                    await websocket.send(d)
                else:
                    buf = b''
                    d = {"data": {"status": 2, "format": format,
                                    "audio": str(base64.b64encode(buf), 'utf-8'),
                                    "encoding": "raw"}}
                    d = json.dumps(d)
                    await websocket.send(d)
                    break
            except Exception as e:
                logging.error("xf send msg exception:", e)
                break
    
    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[str] = None,
    ) -> stt.SpeechEvent:
        buffer = self.change_sample_rate(buffer, 16000)
        buffer: rtc.AudioFrame = agents.utils.merge_frames(buffer)
        duration = len(buffer.data) / buffer.sample_rate
        wsUrl = self.wsParam.create_url()
        now = time.time()
        speechData = MySpeechData(language=language or "zh", text='', start_time=now-duration, end_time=now)

        tasks = []
        async def xf_stt():
            try:
                async with websockets.connect(wsUrl, close_timeout=0.005) as websocket:
                    await self.send_audio(websocket, buffer)
                    message = await websocket.recv()
                    speechData.text = self.parse_message(message)
                    logging.info("xf recognize result:%s" % speechData.text)
                    await websocket.close()
            except Exception as e:
                logging.error("xf recognize exception:", e)
                return stt.SpeechEvent(is_final=True, alternatives=[])
        async def speaker_detect():
            if self.speaker_detector:
                speaker_id = await self.speaker_detector.get_speakerid_from_buffer_async(buffer.data, buffer.sample_rate)
                speechData.speaker_id = speaker_id
        tasks.append(xf_stt())
        if self.speaker_detector:
            tasks.append(speaker_detect())
        await asyncio.gather(*tasks)
        return stt.SpeechEvent(is_final=True, alternatives=[speechData])
    