from livekit.agents import stt
from dataclasses import dataclass, field

@dataclass
class MySpeechData(stt.SpeechData):
    speaker_id: str = ''
