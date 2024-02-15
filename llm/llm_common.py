from .llm_base import LLMBase, VisualLLMBase
from .chatgpt import ChatGPT
from .qwen import Qwen, QwenVisualLLM
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import config

def get_llm() -> LLMBase:
    if config.llm.engine == 'chatgpt':
        logging.info("Using ChatGPT as LLM engine")
        return ChatGPT()
    else:
        logging.info("Using Qwen as LLM engine")
        return Qwen()

def get_vl_LLM() -> VisualLLMBase:
    vlm: VisualLLMBase = QwenVisualLLM() if config.llm.vl_engine == "qwen" else None
    return vlm