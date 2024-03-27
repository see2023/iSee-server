import yaml
from typing import Dict, List

class CommonConfig:
    def __init__(
            self,
            motion_mode: str,
            motion_mode_start_cmd: str,
            motion_mode_cmd_response_ok: str,
            motion_mode_cmd_response_notfound: str,
            motion_mode_stop_cmd: str,
            motion_mode_stop_response: str,
    ):
        self.motion_mode = motion_mode
        self.motion_mode_start_cmd = motion_mode_start_cmd
        self.motion_mode_cmd_response_ok = motion_mode_cmd_response_ok
        self.motion_mode_cmd_response_notfound = motion_mode_cmd_response_notfound
        self.motion_mode_stop_cmd = motion_mode_stop_cmd
        self.motion_mode_stop_response = motion_mode_stop_response

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]):
        return cls(**config_dict)


class AgentsConfig:
    def __init__(
            self,
            show_detected_results: bool,
            yolo_verbose: bool,
            yolo_model: str,  # model: yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt
            yolo_device: str, # device: cpu  cuda mps
            yolo_frame_interval: float,
            stt_type: str, 
            max_buffered_speech: float,
            enable_audio: bool,
            enable_video: bool,
            vision_lang_interval: int,
            min_silence_duration: float,
            min_speaking_duration: float,
            send_vl_result: bool,
            vad_threshold: float,
            log_debug: bool,
            vad_asume_speech_min_prob: float,
            vad_asume_speech_max_count: int,
            speaker_distance_threshold: float,
            min_chunk_duration: float,
            speaker_write_wav: bool,
    ):
        self.show_detected_results = show_detected_results
        self.yolo_verbose = yolo_verbose
        self.yolo_model = yolo_model
        self.yolo_device = yolo_device
        self.yolo_frame_interval = yolo_frame_interval
        self.stt_type = stt_type
        self.max_buffered_speech = max_buffered_speech
        self.enable_audio = enable_audio
        self.enable_video = enable_video
        self.vision_lang_interval = vision_lang_interval
        self.min_silence_duration = min_silence_duration
        self.min_speaking_duration = min_speaking_duration
        self.send_vl_result = send_vl_result
        self.vad_threshold = vad_threshold
        self.log_debug = log_debug
        self.vad_asume_speech_min_prob = vad_asume_speech_min_prob
        self.vad_asume_speech_max_count = vad_asume_speech_max_count
        self.speaker_distance_threshold = speaker_distance_threshold
        self.min_chunk_duration = min_chunk_duration
        self.speaker_write_wav = speaker_write_wav

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]):
        return cls(**config_dict)

class LLMConfig:
    def __init__(
            self,
            engine: str,
            model: str,
            location: str,
            vl_engine: str,
            vl_model: str,
            enable_openai_functions: bool,
            enable_custom_functions: bool,
            custom_functios_output_use_json: bool,
            vl_cmd_catch_pic: str,
            chat_history_count: int,
            chat_history_time_limit: int,
            tts: str,
            voice_name: str,
            vits_url: str,
            split_skip_comma: bool,
            split_min_length: int,
            cache_root:str,
    ):
        self.engine = engine.lower()
        self.model = model
        self.location = location
        self.vl_engine = vl_engine.lower()
        self.vl_model = vl_model.lower()
        self.enable_openai_functions = enable_openai_functions
        self.enable_custom_functions = enable_custom_functions
        self.custom_functios_output_use_json = custom_functios_output_use_json
        self.vl_cmd_catch_pic = vl_cmd_catch_pic
        self.chat_history_count = chat_history_count
        self.chat_history_time_limit = chat_history_time_limit
        self.tts = tts
        self.voice_name = voice_name
        self.vits_url = vits_url
        self.split_skip_comma = split_skip_comma
        self.split_min_length = split_min_length
        self.cache_root = cache_root


    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]):
        return cls(**config_dict)
    
class APIConfig:
    def __init__(
            self,
            url_prefix: str,
            www_root: str,
    ):
        self.url_prefix = url_prefix
        self.www_root = www_root

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]):
        return cls(**config_dict)

class Config:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        all_config: Dict[str, any] = yaml.safe_load(open(config_file, 'r', encoding='utf-8'))
        self.common: CommonConfig = CommonConfig.from_dict(all_config['common'])
        self.agents: AgentsConfig = AgentsConfig.from_dict(all_config['agents'])
        self.llm: LLMConfig = LLMConfig.from_dict(all_config['llm'])
        self.api: APIConfig = APIConfig.from_dict(all_config['api'])


config = Config()
