from .llm_base import LLMBase, VisualLLMBase
from .chatgpt import ChatGPT
from .qwen import Qwen, QwenVisualLLM
from .fine_tune.qwen_local import QwenLocal
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import config

def get_llm() -> LLMBase:
    if config.llm.engine == 'chatgpt':
        logging.info("Using ChatGPT as LLM engine")
        return ChatGPT()
    elif config.llm.engine == 'qwen_local':
        logging.info("Using QwenLocal as LLM engine")
        return QwenLocal(config.llm.model, config.llm.cache_root)
    else:
        logging.info("Using Qwen as LLM engine")
        return Qwen()

def get_vl_LLM() -> VisualLLMBase:
    vlm: VisualLLMBase = QwenVisualLLM() if config.llm.vl_engine == "qwen" else None
    return vlm