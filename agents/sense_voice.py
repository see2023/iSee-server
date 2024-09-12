from typing import Optional
from livekit import agents, rtc
from livekit.agents.utils import AudioBuffer
from livekit.agents import stt
import logging
import torch
import time
import asyncio
from stt_base import MySpeechData
from speaker import Speaker
from agents_tools import memoryview_to_tensor
# import numpy as np
# from funasr_onnx import SenseVoiceSmall
from funasr import AutoModel
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

class SenseVoiceSTT(stt.STT):
    def __init__(self, *, streaming_supported: bool = False, speaker_detector: Speaker = None) -> None:
        super().__init__(streaming_supported=streaming_supported)
        self.speaker_detector:Speaker = speaker_detector

        model_dir = "iic/SenseVoiceSmall"
        # self.model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)
        self.model = AutoModel(model=model_dir, trust_remote_code=False, disable_update=True)
        logging.info("sense_voice stt init success")
    
    @classmethod
    def change_sample_rate(cls, buffer: AudioBuffer, sample_rate: int) -> AudioBuffer:
        if buffer.sample_rate == sample_rate:
            return buffer
        return buffer.remix_and_resample(sample_rate, buffer.num_channels)
    
    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[str] = None,
    ) -> stt.SpeechEvent:
        buffer = self.change_sample_rate(buffer, 16000)
        buffer: rtc.AudioFrame = agents.utils.merge_frames(buffer)
        duration = len(buffer.data) / buffer.sample_rate
        now = time.time()
        speechData = MySpeechData(language=language or "zh", text='', start_time=now-duration, end_time=now)

        tasks = []
        async def sencevoice_stt():
            try:
                # res = self.model(np.frombuffer(buffer.data, dtype=np.int16), language="auto", use_itn=True)

                # expected Tensor as element 0 in argument 0, but got memoryview
                input = memoryview_to_tensor(buffer.data)
                res = self.model.generate(
                    input=input,
                    cache={},
                    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=True,
                )
                speechData.text = rich_transcription_postprocess(res[0]["text"])
                logging.info("sense_voice recognize result:%s" % speechData.text)
            except Exception as e:
                logging.error("sense_voice recognize exception:", e)
                return stt.SpeechEvent(is_final=True, alternatives=[])
        async def speaker_detect():
            if self.speaker_detector:
                speaker_id = await self.speaker_detector.get_speakerid_from_buffer_async(buffer.data, buffer.sample_rate)
                speechData.speaker_id = speaker_id
        tasks.append(sencevoice_stt())
        if self.speaker_detector:
            tasks.append(speaker_detect())
        await asyncio.gather(*tasks)
        return stt.SpeechEvent(is_final=True, alternatives=[speechData])
    