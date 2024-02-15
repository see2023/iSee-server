import io
import wave
from typing import Optional
from livekit import agents
from livekit.agents.utils import AudioBuffer
from livekit.agents import stt
from faster_whisper import WhisperModel as FastWhisperModel
import whisper
import logging
import time
import numpy as np
import torch


class STT(stt.STT):
    def __init__(
            self,
            *,
            use_fast_whisper: bool = True,
            language: str = "en",
            model_size_or_path: str = "large-v3", 
                    # fast: #tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
                    # original:  #['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
            device: str = "auto", # fast: cpu,cuda,auto
            compute_type: str = "default", #fast:  int8 int8_float32 int8_float16 int8_bfloat16 int16 float16 bfloat16 float32
            beam_size: int = 5,
            initial_prompt: str = "你好！",
    ):
        super().__init__(streaming_supported=False)
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.language = language
        self.use_fast_whisper = use_fast_whisper
        logging.info("loading whisper path: %s", model_size_or_path)
        if use_fast_whisper:
            self._model = FastWhisperModel(
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type=compute_type,
            )
            logging.info("fast whisper loaded.....")
        else:
            self._model = whisper.load_model(
                model_size_or_path,
                device=device,
            )
            logging.info("original whisper loaded.....")

    async def recognize(
        self,
        *,
        buffer: AudioBuffer,
        language: Optional[str] = None,
    ) -> stt.SpeechEvent:
        buffer = agents.utils.merge_frames(buffer)
        logging.info("transcribing")
        start_time = time.time()

        text = ""
        detect_language = ""
        duration = 0
        try:
            if self.use_fast_whisper:
                io_buffer = io.BytesIO()
                with wave.open(io_buffer, "wb") as wav:
                    wav.setnchannels(buffer.num_channels)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(buffer.sample_rate)
                    wav.writeframes(buffer.data)
                io_buffer.seek(0)

                segments,info =  self._model.transcribe(io_buffer,
                        beam_size=self.beam_size,
                        initial_prompt=self.initial_prompt,
                        language=self.language,
                    )
                for segment in segments:
                    t = segment.text
                    if t.strip().replace('.', ''):
                        text += ', ' + t if text else t
                duration += info.duration
                detect_language = info.language

            else:
                numpy_array = np.array(buffer.data)
                audio_float32 = numpy_array.astype(np.float32)
                audio_normalized = audio_float32 / np.iinfo(np.int16).max
                audio_tensor = torch.tensor(audio_normalized)

                rt =  self._model.transcribe(audio_tensor,
                        beam_size=self.beam_size,
                        initial_prompt=self.initial_prompt,
                        language=self.language,
                )
                text = rt['text']
                detect_language = rt['language']
                for segment in rt['segments']:
                    if segment['end'] and segment['start']:
                        duration += segment['end'] - segment['start']
        
        except Exception as e:
            logging.error("failed to transcribe: %s", e)
            return transcription_to_speech_event(stt.SpeechData(text="", language=""))
        logging.debug("transcribed -------------: %s, time used: %.3fs", text, time.time() - start_time)
        speech_data = stt.SpeechData(text=text, language=detect_language, start_time=start_time, end_time=start_time+duration)

        return transcription_to_speech_event(speech_data)

def transcription_to_speech_event(transcription) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        is_final=True,
        alternatives=[transcription],
    )
