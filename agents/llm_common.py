from enum import Enum
from dataclasses import dataclass
import re

class MessageType(Enum):
    TEXT = "text"
    VISUAL_ANALYSIS_REQUEST = "visual_analysis_request"
    VISUAL_ANALYSIS_RESULT = "visual_analysis_result"
    FOLLOW_UP_QUESTION = "follow_up_question"

@dataclass
class LLMMessage:
    type: MessageType
    content: str

    def __str__(self):
        # Custom string format without enum value display
        return f"LLMMessage(type=MessageType.{self.type.name}, content='{self.content}')"

    @classmethod
    def from_string(cls, s: str):
        # Regex to extract type and content from the string
        match = re.match(r"LLMMessage\(type=MessageType\.(\w+), content='(.*)'\)", s)
        if not match:
            raise ValueError("String format is incorrect")

        message_type_str, content = match.groups()
        message_type = MessageType[message_type_str]  # Convert to MessageType enum

        return cls(type=message_type, content=content)

# Example usage
VISUAL_ANALYSIS_REQUEST = LLMMessage(MessageType.VISUAL_ANALYSIS_REQUEST, "请进行视觉分析")
