# from livekit.rtc ChatManager, ChatMessage, but has custom srcname,timestamp,duration,language

from dataclasses import dataclass, field
import time
import json
import logging
import asyncio
from typing import Any, Callable, Dict, Literal, Optional

from livekit.rtc.room import Room, Participant, DataPacket
from livekit.rtc._event_emitter import EventEmitter
from livekit.rtc._proto.room_pb2 import DataPacketKind
from livekit.rtc._utils import generate_random_base62

import msgpack

_CHAT_TOPIC = "lk-chat-topic"
_AUDIO_TOPIC = "lk-audio-topic"
_MOVE_TOPIC = "lk-move-topic"
_CHAT_UPDATE_TOPIC = "lk-chat-update-topic"

CHAT_MEMBER_APP = "app"
CHAT_MEMBER_ASSITANT = "assistant"
CHAT_MEMBER_WEB = "web"
CHAT_MEMBER_STT = "stt"
CHAT_MEMBER_MSG_ARRANGE = "arrange"

EventTypes = Literal["message_received", "move_received"]

class ChatExtManager(EventEmitter[EventTypes]):
    """A utility class that sends and receives chat messages in the active session.

    It implements LiveKit Chat Protocol, and serializes data to/from JSON data packets.
    """

    def __init__(self, room: Room):
        super().__init__()
        self._lp = room.local_participant
        self._room = room

        room.on("data_received", self._on_data_received)

    def close(self):
        self._room.off("data_received", self._on_data_received)

    async def send_message(self,message: str,  srcname: Optional[str] = None, timestamp: Optional[float] = None,duration: Optional[float] = None,language: Optional[str] = None) -> "ChatExtMessage":
        """Send a chat message to the end user using LiveKit Chat Protocol.

        Args:
            message (str): the message to send

        Returns:
            ChatMessage: the message that was sent
        """
        msg = ChatExtMessage(
            message=message,
            is_local=True,
            participant=self._lp,
            timestamp=timestamp,
            duration=duration,
            language=language,
            srcname=srcname,
        )
        await self._lp.publish_data(
            payload=json.dumps(msg.asjsondict()),
            # reliable=True,
            # kind=DataPacketKind.KIND_RELIABLE,
            topic=_CHAT_TOPIC,
        )
        return msg
    
    async def send_chunk(self, chunk_data, topic, semaphore: asyncio.Semaphore, user_sid: str):
        async with semaphore:
            try:
                await self._lp.publish_data(
                    payload=chunk_data,
                    # reliable=True,
                    # kind=DataPacketKind.KIND_RELIABLE,
                    topic=topic,
                    destination_sids=[user_sid],
                )
            except Exception as e:
                logging.warning("Failed to send chunk: %s", e, exc_info=True)

    async def send_audio_message(self, audio_data: bytes, visemes, text_id: str, visemes_fps: float, user_sid: str):
        """Send a chat message to the end user using LiveKit Chat Protocol.

        Args:
            audio_data (bytes): the audio data to send
            visemes (list): the visemes data to send
            text_id (str): the text id of the response text

        Returns: msg id:str
        """
        all_data = {
            "text_id": text_id,
            "visemes": visemes,
            "audio_data": audio_data,
            "visemes_fps": visemes_fps,
        }
        payload_all = msgpack.packb(all_data, use_bin_type=True)
        # slit the payload into 14k each
        total_size = len(payload_all)
        chunk_size = 14000
        if len(payload_all) % chunk_size == 0:
            chunk_count = total_size // chunk_size
        else:
            chunk_count = total_size // chunk_size + 1
        id = generate_random_base62()
        tasks = []
        semaphore = asyncio.Semaphore(10)
        try:
            for i in range(chunk_count):
                topic = f"{_AUDIO_TOPIC}/{id}/{chunk_count}/{i+1}"
                start = i * chunk_size
                end = min(start + chunk_size, total_size)
                chunk_data = payload_all[start:end]
                tasks.append(self.send_chunk(chunk_data, topic, semaphore, user_sid))
            await asyncio.gather(*tasks)
            return id 
        except Exception as e:
            logging.warning("failed to send audio message: %s", e, exc_info=e)
            return None

    async def send_move_cmd(self, cmd: str, srcname: Optional[str] = None, timestamp: Optional[float] = None,duration: Optional[float] = None,language: Optional[str] = None):
        """

         {"id": "CtoVAxPJ8Tyu", "timestamp": 1706932676704, "srcname": "assistant","cmd": "A"}
        """
        try:
            msg = ChatExtMessage(
                cmd=cmd,
                participant=self._lp,
                timestamp=timestamp,
                duration=duration,
                language=language,
                srcname=srcname,
            )
            await self._lp.publish_data(
                payload=json.dumps(msg.asjsondict()),
                # reliable=True,
                # kind=DataPacketKind.KIND_RELIABLE,
                topic=_MOVE_TOPIC,
            )  
        except Exception as e:
            logging.warning("failed to send move cmd: %s", e, exc_info=e)
            

    async def update_message(self, message: "ChatExtMessage"):
        """Update a chat message that was previously sent.
        """
        await self._lp.publish_data(
            payload=json.dumps(message.asjsondict()),
            reliable=True,
            topic=_CHAT_UPDATE_TOPIC,
        )

    def on_message(self, callback: Callable[["ChatExtMessage"], None]):
        """Register a callback to be called when a chat message is received from the end user."""
        self._callback = callback

    def _on_data_received(self, dp: DataPacket):
        # handle both new and updates the same way, as long as the ID is in there
        # the user can decide how to replace the previous message
        if dp.topic == _CHAT_TOPIC or dp.topic == _CHAT_UPDATE_TOPIC:
            try:
                parsed = json.loads(dp.data)
                msg = ChatExtMessage.from_jsondict(parsed)
                if dp.participant:
                    msg.participant = dp.participant
                self.emit("message_received", msg)
            except Exception as e:
                logging.warning("failed to parse chat message: %s", e, exc_info=e)
        elif dp.topic == _MOVE_TOPIC:
            try:
                parsed = json.loads(dp.data)
                msg = ChatExtMessage.from_jsondict(parsed)
                if dp.participant:
                    msg.participant = dp.participant
                self.emit("move_received", msg)
            except Exception as e:
                logging.warning("failed to parse move message: %s", e, exc_info=e)

@dataclass
class ChatExtMessage:
    id: str = field(default_factory=generate_random_base62)
    timestamp: float = time.time
    message: Optional[str] = None
    srcname: Optional[str] = None
    deleted: bool = field(default=False)
    participant: Optional[Participant] = None
    is_local: bool = False

    duration: Optional[float] = 0
    language: Optional[str] = 'zh'
    cmd: Optional[str] = None

    def asjsondict(self) -> Dict[str, Any]:
        if id is None:
            self.id = generate_random_base62()
        if self.timestamp is None:
            self.timestamp = time.time()
        
        return {
            "id": self.id,
            "timestamp": int(self.timestamp*1000),
            "message": self.message,
            "srcname": self.srcname,
            "deleted": self.deleted,
            "is_local": self.is_local,
            "duration": self.duration,
            "language": self.language,
            "cmd": self.cmd,
        }

    def update_from_jsondict(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    @staticmethod
    def from_jsondict(data: Dict[str, Any]) -> "ChatExtMessage":
        id = data.get("id") or generate_random_base62()
        timestamp = time.time()
        if data.get("timestamp"):
            timestamp = data.get("timestamp", 0) / 1000.0
        return ChatExtMessage(
            id=id,
            timestamp=timestamp,
            message=data.get("message", ""),
            srcname=data.get("srcname", ""),
            is_local=data.get("is_local", False),
            duration=data.get("duration", 0),
            language=data.get("language", "zh"),
            cmd=data.get("cmd", ""),
        )

    def getname(self) -> str:
        return self.srcname or self.participant.name
